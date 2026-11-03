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

| Hint, and where it was placed | Total cost |
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

| Measured on out-of-bag rows | log-ratio units |
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

| Conformal calibration | value |
|---|---|
| alpha | 0.10 |
| **conformal multiplier q** | **8.51** |
| calibration rows | 229 |
| held-out coverage (target 0.90) | 1.00 |

**q = 8.51 against an assumed z = 1.0.** Covering 90% of outcomes takes eight
and a half sigmas, not one. The reason is in the same eval output:

| Spread against error | log-ratio units |
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

**And the JOB column must not be read as a JOB result.** `app.onboard` generated
those queries, and native averages **210.9 ms** on them. The actual Join Order
Benchmark queries on the same database average **3703 ms** native against a best
available plan of 919 ms, and against those this system captures **0%** of the
75% gap -- every selector returns the native plan (`docs/JOB_RESULTS.md`). The
+77.2% below is a real measurement of a real difficulty; it is not the
difficulty JOB is famous for. Quoting it without that sentence is how a project
ends up believing its own easiest number, which is the failure this section's
whole method exists to avoid.

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

| Path taken | Cumulative regret |
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


## 2.20 Asking the planner instead of inferring from cost

Four times now this project has been bitten by the same thing, and §2.0 was the
worst of them: pg_hint_plan silently ignoring every hint, which made every
"candidate" byte-identical to native and every measured improvement timing
noise. The others:

* `Parallel(order_items 4 hard)` named the table where the query said `oi`.
  Cost 7870, unchanged. `Parallel(oi 4 hard)` cost 1774.
* `DisableIndex` does not exist in pg_hint_plan 1.6.3 — the PG18 manual
  documents a newer one.
* Two `/*+ ... */` comments: only the first is read, the rest vanish.

Every one is the same shape. **A hint that is not applied is a SQL comment**:
no error, no warning, and a plan identical to the one you would have got anyway.
The defence so far has been indirect — compare the hinted plan's cost against
native and assume a difference means the hint bound. That test has a hole big
enough to drive all four bugs through, because it cannot distinguish

* a hint that bound and legitimately changed nothing, from
* a hint that never bound at all.

Both leave the cost untouched. On the four-table query, `SeqScan(o)` and
`SeqScan(orders)` both cost **3979.3**.

### The obvious readback does not work

`EXPLAIN (SETTINGS)` reports planner GUCs differing from their built-in
defaults, which sounded like a direct answer for the `Set()` family — by
measured value the most useful family in the action space (§2.19). It is not:

| | cost | `Settings` |
|---|---|---|
| `SET enable_seqscan = off` | 1.0000039e10 | `{'enable_seqscan': 'off'}` |
| `/*+ Set(enable_seqscan off) */` | 1.0000039e10 | `{}` |

Identical cost effect, so the hint plainly bound. The option works; pg_hint_plan
applies the GUC in a way that is invisible to it. Worth recording as a negative
result, because it is the first thing anyone would reach for.

### Asking pg_hint_plan directly

pg_hint_plan will simply say. With `debug_print` on it emits one NOTICE per
statement sorting the directives into `used` / `not used` / `duplication` /
`error`, and that covers every family rather than just `Set()`:

| case | reported as |
|---|---|
| `SeqScan(o)` — correct alias | `used hint` |
| `SeqScan(orders)` — table name | **`not used hint`** |
| `Set(bogus_guc off)` | **`error hint`** |
| `DisableIndex(o)` | `hint syntax error` / `Unrecognized hint keyword` |
| two hint comments | only the first appears; the second is in no bucket |

There is a severity trap inside this, and it is why `DisableIndex` looked like
silence for so long: hint **syntax errors arrive at `INFO`**, not at the
configured `message_level`, and the default `client_min_messages` of `WARNING`
discards INFO. The extension was reporting the problem the whole time to a
channel nobody was listening on.

`optimizer/hint_binding.py` parses this into a `BindingReport` with a `bound`
property and a `problems` list. Its `bound` is deliberately false when **no**
debug output arrived: the failure mode being defended against is a missing
message, so "I could not tell" must never read as "it bound".

### What it found in the first hour

Pointed at the full action space — 92 candidates over every family — **9 did
not bind**, and all 9 came from one family.

`generate_join_method_candidates` names the growing prefixes of a join order:
`Leading(p o oi u)` becomes `HashJoin(p o)`, `HashJoin(p o oi)`, and so on. It
sampled its orders with no reference to the join graph. `generate_join_order_candidates`
had taken a join graph for exactly this reason, and its docstring spells out
why: an order that introduces a table sharing no join predicate forces a cross
product, priced at `disable_cost`.

The method generator never got the same treatment, so it could emit
`HashJoin(p o)` for two relations with no predicate between them — `p` and `o`
are two hops apart in `o–oi–p`. pg_hint_plan filed it under `not used hint:`
and planned that join however it liked.

That is worse than the wasted candidate the order-only generator avoids. The
candidate still executes, still gets a latency, and is still labelled *"hash
join throughout"* while its first join was chosen freely. It is a **mislabelled
training row** — and mislabelled in a way correlated with the family, which is
precisely the kind of contamination §2.18 traced from the training set to a gate
that could not fire.

The fix extracts the connected-order selection both generators should have
shared into `_choose_orders`. **9 of 92 → 0 of 93.**

### Then the same check over the whole workload

One query is not evidence about an action space. Run across all 24 workload
queries, generating candidates from each query's real aliases and join graph:

| action space | candidates | did not bind |
|---|---|---|
| **served** (join order + hint sets) | 440 | **0** |
| **full** (every family switched on) | 2102 | **8** |

The first row is the one that matters, and it is the reassuring one: everything
this optimizer actually serves, across every query in the workload, binds.

The 8 were all one shape, and a different bug from the join-method one.
pg_hint_plan's nested `Leading()` takes exactly **two** elements per group:

    Leading(((o u) (oi p ps)))     -> error hint:   rejected
    Leading(((o u) ((oi p) ps)))   -> used hint:    applied

`generate_bushy_candidates` split the tables at `len(tables) // 2`, which is 2
and 3 for a five-table query, and emitted the flat three-element group. Rejected,
at a severity nothing was listening on, so the candidate ran and did nothing.

Fixing it meant more than inserting parentheses. `((a b) c)` joins `a` to `b`
first, so folding a group alphabetically can pair two tables with no predicate
between them -- syntactically fine, priced at `disable_cost`, useless. The fold
follows the join graph instead. **8 of 2102 -> 0 of 2102.**

Both families this found -- forced join methods and bushy trees -- are opt-in and
off at serving, which is why neither had shown up in any benchmark. They are
collected, though, and `collect_data` gathers join methods by default. A defect
that only touches data you are not serving is still a defect in the data you
train on, and 2.18 is the record of what that costs.

One more thing the audit corrected, in the checker rather than the checked: the
`error` bucket's message read *"an unknown GUC or an invalid value"*, written
when the only rejection anyone had seen was a bad `Set()`. The first real
rejection it caught was a malformed `Leading()`. A diagnostic that has learned
only one failure will confidently mislabel the second.

### The general lesson

The check that had been in place — "the hinted plan differs from native, so the
hint worked" — is an *inference* from a side effect. The extension knew the
answer and would have said so for the cost of one GUC. Four bugs, one of them
invalidating an entire round of results, came from inferring what could have
been asked.

Where a component can report its own behaviour, read that report. Do not
reconstruct it from downstream effects, however reasonable the reconstruction
looks — the reconstruction is silent exactly when it is wrong.



## 2.21 Three things wrong with the gate, and only one of them was the gate

Driving the real endpoint over all 24 workload queries, rather than replaying
rows, produced **0 deviations**. The calibration had just measured 56% on the
same model. Two numbers that far apart mean the thing being calibrated is not
the thing being run.

### 1. A ceiling that passed through zero

The downside check read:

```python
if upper_ratio >= self.max_regression_ratio / caution:
```

with a comment directly above it saying this is *"a ceiling on how bad this can
get, not a demand that the worst case also be a win"*.

`caution` is **2.0** for a query with no served history
(`regression_guard.UNSEEN_QUERY_CAUTION`), easing to 1.0 as evidence arrives.
So the ceiling became `1.25 / 2.0 = 0.625`: the pessimistic case had to be a
**1.6x speedup**. That is exactly the demand the comment disclaims, and every
query on a fresh workload starts unseen.

Dividing a ratio only behaves while the result stays above 1.0. Scaling the
*margin* above 1.0 does what the docstring promises:

```python
effective_ceiling = 1.0 + (self.max_regression_ratio - 1.0) / caution
```

| caution | old ceiling | new ceiling |
|---|---|---|
| 1.0 | 1.250 | 1.250 |
| 2.0 | **0.625** | 1.125 |

Monotone, never below 1.0, and still stricter with less evidence.

The test that covered this asserted the old behaviour: a prediction of 0.82x
with a spread of 0.05, refused at caution 2.0. Its pessimistic case is **0.906**
-- the worst plausible outcome was a 9% speedup, and the gate refused it. The
test now uses a case that is genuinely risky rather than merely uncertain (a
27% pessimistic regression), plus one asserting the ceiling can never fall below
1.0 at any caution.

### 2. The payload reported a number the gate never used

The decision carried `max_regression_ratio: 1.25` beside
`pessimistic_speedup_vs_native: 1.152` and the verdict `downside_not_bounded`.
1.152 is less than 1.25. Read literally, the gate vetoed a plan that passed --
and the comment on that field says it is *"reported so a reader can see which of
the two conditions a held plan failed"*.

It was reporting the raw setting while comparing against `setting / caution`.
The payload now carries `effective_max_regression_ratio` and `caution`, and the
dashboard's explanation is the arithmetic that actually ran.

### 3. The calibration does not model caution at all

`caution` does not appear in `calibrate.py`. The sweep replays logged rows and
evaluates thresholds at an implicit caution of 1.0; serving applies 1.33-2.0.
So the recommended `min_relative_gain=0.02` is really 0.04 at serving, and the
1.25 ceiling is really 1.125.

This is the same shape as 2.18 -- calibrate on one distribution, serve another --
one level up. **A gate tuned without the multiplier the gate applies is tuned
for a system that does not exist.**

### What the fix was actually worth, and what it was not

End-to-end deviations went from **0/24 to 2/24**. The remaining 22 split evenly:
11 `no_expected_gain_over_native`, 11 `downside_not_bounded`.

The honest reading is that caution was not the main constraint. Measured
pessimistic ratios on the vetoed queries:

    2.226  1.433  1.425  1.555  1.577  1.371  1.253  1.160  1.016

Most exceed **1.25** -- the uncaution'd ceiling. Raising the ceiling to any
defensible value does not admit them. The bands are that wide because
`uncertainty_scale` is **3.88**, fitted to reach the target coverage.

So the gate is not miscalibrated so much as the model's uncertainty is too wide
to authorise most deviations, and that is a statement about the model, not the
threshold. It is the same conclusion 2.19's refusal message was rewritten to
deliver, arrived at from the serving side: **collect more data.**

### Fixing the third one changed the answer

Making `calibrate` apply the caution the gate applies moved its recommendation
from `z=2.0, min_gain=0.02` to **`z=0.5, min_gain=0.0`** -- lower on both,
because serving doubles the z-margin and doubles the required gain, and the
sweep now compensates instead of being surprised.

End-to-end deviations, driving the real endpoint over all 24 queries:

| gate | deviated | `downside_not_bounded` |
|---|---|---|
| original (`z=2.0, gain=0.20`), ceiling `/caution` | **0 / 24** | 11 |
| ceiling fixed, `z=2.0, gain=0.02` | 2 / 24 | 11 |
| ceiling fixed, caution-aware `z=0.5, gain=0.0` | **16 / 24** | **0** |

Every remaining veto is `no_expected_gain_over_native` -- the model saying it
sees nothing worth doing, which is the honest refusal. None is the gate refusing
a plan the model liked.

And executed, not predicted (`app.benchmark`):

| | native | served | oracle | headroom captured |
|---|---|---|---|---|
| all 24 | 3155.4 ms | **2669.7 ms** | 1915.2 ms | **39.2%** |
| 9 held-out | 713.8 ms | **628.5 ms** | -- | **22.0%** |

**485 ms, 15.4% faster overall**, with 7 safety vetoes and 3 guard blocks still
firing. The held-out row is the one that matters: 22% of available headroom on
queries the model never trained on, against **0.4%** measured for the deployed
gate at the start of this section.

Two caveats worth keeping attached to those numbers, and 2.26 turns the first
into something sharper. They are a **single run**, and 2.18 records that
repeated executions of the same plan differ by 18.6% on average -- so 39.2% is
not precise to its third digit. And the database has changed since the model was
trained, which is the hazard that section was written about.

**Superseded.** This run was taken immediately after applying a gate, which
2.26 shows is the most pessimistic moment in the system's life: the guard's
evidence window is empty, every query is unseen, and caution is at its maximum
of 2.0. Five consecutive runs of a later configuration moved from 8.4% to 46.3%
on held-out queries with nothing changed but accumulated evidence. Read 39.2% /
22.0% as a cold-start measurement of that configuration, not as its steady
state, and read the medians in 2.26 as the headline.

Which does not make the three fixes cosmetic. A ceiling below 1.0 was
unreachable at any confidence, a payload that cannot explain its own verdict
costs every future debugging session, and a calibration blind to a factor-of-two
multiplier will keep recommending settings that mean something else in
production. They were all found by driving the real endpoint instead of trusting
the replay -- which is 2.20's lesson again, one layer out.



## 2.22 Three places where work was being done and thrown away

Not bugs in the sense of wrong answers. Each of these produced correct output
and paid for something it did not keep.

### The candidate budget was funding duplicates

`plan_query` sliced the hint list to the budget and *then* deduplicated the
plans that came back:

```python
hints = hints[:budget]      # spend the budget
...
if fingerprint in seen_plans:
    continue                # discover what it bought
```

Many hints legitimately collapse to the same plan -- on a small join PostgreSQL
answers most `Leading()` orders with the plan it already preferred. Measured on
a four-table join, **56 of 83** candidates were duplicates of one another. The
budget was funding repeats while genuinely different alternatives further down
the list were never planned at all.

The budget now counts *distinct plans*, capped at
`MAX_PLANNING_ATTEMPTS_PER_SLOT` tries per slot so a fully degenerate query --
every hint collapsing to the baseline -- cannot plan the entire list looking for
something that is not there. Planning one hint is an `EXPLAIN` without
`ANALYZE`, a few milliseconds, so trying a few extra to fill the budget with
real alternatives is cheap. The decision payload also reports
`n_duplicate_plans`, because "this query has few alternatives" and "this query's
alternatives all collapse" are different facts and only the second argues for
generating a different *kind* of candidate.

### Collection was measuring a family training would discard

2.18 fixed the model by dropping rows for families the serving path does not
offer, and made `train.py` read *which* families from `planner.py` so the two
could not drift. Collection was the one place still asserting an answer of its
own:

```python
def collect(reps=DEFAULT_REPS, include_join_methods: bool = True, ...)
```

`INCLUDE_JOIN_METHODS` is off. So every collection run spent most of its wall
clock measuring the forced join-method family -- the **slowest** in the action
space, mean ratio 5.49 against 1.97 for everything else, and multiplied by the
repetition count -- and `rows_for_served_action_space` then discarded every row
of it.

Measured directly: a 5-repetition run was still grinding through forced merge
joins after an hour, having produced 2,465 rows that would not survive the
filter. Defaulting the parameter to `planner.INCLUDE_JOIN_METHODS` instead took
the rate from **41 rows/minute to about 100**, every one of them a row the model
will actually train on. Passing the flag explicitly still works, which is what a
run meant to answer *"would join methods earn their place?"* should do.

The pattern is worth naming, because this is the third time it has appeared:
**a constant that duplicates a decision made elsewhere will eventually disagree
with it.** 2.18 was the model trained on the wrong action space, 2.21 was
calibration tuning a gate the server does not run, and this is collection
measuring a space training discards. In each case the fix was the same shape --
read the decision from wherever it is actually made.

### The advisor knew about one index type

`dbadvisor` proposed B-tree for everything except the trigram case, and BRIN is
core -- no extension needed. One column in this schema is exactly its shape:
`orders.created_at` on an append-only table.

`index_advice.brin_opportunities` reads `pg_stats.correlation`, which PostgreSQL
already computes during `ANALYZE`, rather than probing for it. That matters for
the failure mode: BRIN summarises min/max per *page range*, so on a well-ordered
column a range scan skips whole swathes of the table with an index a few hundred
kilobytes where the B-tree is tens of megabytes -- and on a poorly-ordered one
every range matches and it degenerates into a sequential scan plus wasted index
reads.

So a false positive costs more than staying silent, and the rule is
correspondingly strict: correlation at least 0.9, at least 2,000 pages, and only
types where a min/max summary discriminates (a BRIN on a boolean summarises to
"contains both" in every range, however perfectly correlated). A column that has
never been analyzed has no `pg_stats` row and gets no advice, rather than advice
built on a guess.




## 2.23 The model registry was reporting a decision it did not control

`model_store.py` exists to stop the pipeline from overwriting the served model
on every train. Its opening docstring says so: *"Until now the pipeline
overwrote `models/plan_selector.pkl` on every train"*. `retrain.py` states the
invariant where it depends on it:

> `train()` writes to a scratch path -- promotion is this module's decision, not
> `train()`'s.

And it honours that, by passing `models/_challenger.pkl` explicitly.

`train.py`'s default did not:

```python
MODEL_PATH = os.getenv("MODEL_OUT_PATH", "models/plan_selector.pkl")
```

That path is `model_store.CURRENT_PATH` -- the file `learned.py` opens to serve
with. So `python -m app.train`, the documented way to train from a terminal,
wrote the new model straight onto the served one. The registry was never
consulted, and went on reporting the version it had last promoted.

### Caught by accident, which is the point

Retraining on a fresh collection produced a model whose offline evaluation was
worse than the incumbent on every policy -- risk-averse headroom **0.045**
against **0.365**. The registry correctly refused to promote it. Checking that
the refusal had taken effect:

    plan_selector.pkl : 7d81ed5bbc2a5147
    registry current  : acfc6ca7bf571754 -> 20260902T123233Z
    MISMATCH -- serving a model the registry did not promote

The pickle had already landed. The refusal was real, recorded, and completely
without effect, and nothing anywhere reported a problem: `/model/status` read
the registry and named the promoted version, while a different model answered
every query. This is the same silent shape as 2.20 and 2.21 -- a mechanism that
reports correctly while being bypassed -- and it is now three layers deep:
hints that did not bind, a gate ceiling that could not be met, and a promotion
that did not deploy.

### The fix, and where the check belongs

`train()` now refuses a `model_path` that resolves to the served file, naming
the route that does make the decision. Refused loudly rather than silently
redirected: a run that believes it deployed a model and did not is the same
failure pointing the other way.

The guard sits at the *top* of `train()`, not beside the write it protects.
That was not where it started -- putting it next to the write meant an invalid
destination was reported only after a full training run, which the tests
noticed by taking 14 seconds to assert an error. They now take 0.05.

`test_train_does_not_deploy.py` holds both halves, because either alone is
worth nothing: that the CLI default is not the served path, **and** that the
served path is still the file `learned.py` loads. If `CURRENT_PATH` ever stops
being that file, the first check would pass while protecting nothing.

### And then the model it had refused turned out to be better

The offline evaluations that condemned the challenger are computed on different
samples -- 2.18 records native latency moving 247 -> 184 -> 205 ms across
collections -- so comparing two of them compares two different worlds.

`retrain.py` already had the honest instrument: score champion and challenger
on a **common** held-out set. Run that way:

    champion   178.59 ms
    challenger 163.74 ms
    improvement 8.3%, above the 2% bar -> promoted

The challenger was 8.3% better, on the only comparison where both models answer
the same questions. The offline eval had said worse; the paired comparison says
better; the paired comparison is the one that means anything.

Worth stating plainly, because it cuts against the section above: the bug let a
model through that turned out to be an improvement. That is not a defence of
the bug. A gate that admits a good model by accident is not a gate, and the next
model it admits is not chosen either.



## 2.24 Where C++ would help: measured, and the answer is nowhere

The question was whether part of this system should be rewritten in C++. It is
the kind of question that is usually answered by taste, so it was answered by
measurement instead -- three of them, and they all point the same way.

### What the optimizer's overhead is actually made of

Measured across all 24 workload queries through the real endpoint:

| measurement | value |
|---|---|
| optimizer overhead, median | **26.3 ms** |
| optimizer overhead, worst | 58.2 ms |
| a query's own execution, median | ~110 ms |

So the overhead is real -- about a quarter of what the query itself costs. Then
what is inside it, for one six-table query with 18 candidates:

| | | |
|---|---|---|
| one `EXPLAIN` at a time (what it does) | **17.28 ms** | |
| planning floor (`EXPLAIN` x 18) | **13.76 ms** | **80%** |
| bare round trip x 18 | 3.60 ms | 21% |
| feature extraction x 18 | 2.43 ms | 14% |

**Eighty percent of it is PostgreSQL planning**, which is the work being asked
for and is already C. What remains is a few milliseconds of round trip and a
couple of milliseconds of Python arithmetic, against a query that takes 110 ms.

A C++ rewrite of the feature extractor -- the obvious candidate, since it is the
only real arithmetic in the serving path -- would target **2.4 ms of a 26 ms
overhead on a 110 ms query**: about 2% of the end-to-end time, in exchange for a
build toolchain, a native dependency and a second implementation of a function
that is currently 90 lines of readable Python. The optimizer's *purpose* is to
find plan improvements worth 15-40%.

### The idea that looked best was three times slower

The one genuinely appealing design was a PostgreSQL extension: plan every
candidate inside the server, so 18 round trips become 1. The image already has
`build-essential` and `postgresql-server-dev-16` -- it builds pg_hint_plan and
HypoPG from source -- so the toolchain was not the obstacle.

Before writing it, the same saving was measured without it, by sending all 18
`EXPLAIN`s as one multi-statement call:

    one EXPLAIN at a time      17.28 ms
    all in one round trip      51.95 ms

**Three times slower.** Whatever the server does with a multi-statement message,
it is not what a loop of single statements gets, and an extension would have had
to beat the 17.28 ms rather than the 51.95 ms it might have been credited with.
Its ceiling was the 3.6 ms of round trips -- 21% of the overhead, 3% of the
query.

### The answer, and what would change it

There is no CPU-bound hot spot in this system. It is bound by the database and
by measurement noise, and both are outside anything a language choice reaches.
C++ is not warranted anywhere in it today.

Three things would change that answer, and none is true here:

* **A workload of short queries.** At 110 ms per query a 26 ms overhead is
  tolerable; at 5 ms it dominates, and then the round trips and the Python are
  most of what is left after planning.
* **A network between the optimizer and the database.** Round trips cost 0.2 ms
  to a container on the same machine. At 1 ms they become the largest removable
  cost, and the extension idea comes back on merit.
* **Serving without a Python runtime.** LightGBM's predictor is C++ already, and
  a native scorer would let the optimizer live inside another process. That is a
  deployment capability rather than a speedup, and nothing here has asked for it.

Recorded because the measurement is the useful artifact. The next person to ask
"should this be in C++" can read three numbers instead of re-deriving them, and
knows which of them to re-measure if their workload differs.



## 2.25 The regression guard was blocking on evidence about a different system

After the new model was promoted and the gate recalibrated by measurement, the
benchmark reported:

    safety vetoes: 3/24   guard-blocked: 5/24

The guard was stopping more queries than the safety gate. It is supposed to be
the *retrospective* half of the protection -- a query that has actually come out
slower gets stopped regardless of what the model predicts -- so five blocks
against a freshly promoted model is a strange number.

Its SQL says why:

```sql
SELECT query_id, AVG(...) FILTER (WHERE is_baseline), ...
FROM plan_execution_log
WHERE query_id IS NOT NULL AND actual_total_time_ms IS NOT NULL
GROUP BY query_id
```

No date filter, no model filter, no gate filter. **Every execution ever logged
counts, forever, with equal weight.** So the guard was blocking the current
model for regressions produced by a superseded one -- specifically by the gate
whose ceiling collapsed to 0.625 (2.21) and by the replay-chosen setting that
measured -3.9% on held-out queries.

Every number in those blocks was true. All of them were about a system that no
longer exists.

### The same sentence, one layer over

2.18 ends with: *"a plan-latency log is only training data while the database it
measured is still the database you are serving."* This is that applied to the
evidence a guard blocks on, and the failure is silent in the same way -- the
guard reports a blocked query with a real regression ratio computed from real
executions, and is confidently protecting against the past.

### The window, and where it comes from

The horizon is the **current model's promotion time**, because that is the event
that makes older evidence describe something else. `model_store.promoted_at()`
already had the timestamp; it just had no accessor.

Two properties matter more than the choice of horizon:

* A very old promotion does not widen the window forever -- `DEFAULT_LOOKBACK`
  caps it, so a model promoted a year ago cannot resurrect a year of evidence
  gathered across databases that have since changed.
* An unreadable registry **narrows** the window to the fixed lookback rather
  than falling back to all of history. The dangerous default is the permissive
  one.

A freshly promoted model therefore starts with no history and no blocks. That is
correct: it has not earned distrust yet, and `caution_multiplier` separately
handles the fact that it has not earned trust either.

Measured after the change: **guard-blocked 5/24 -> 2/24.**

### Two bugs in the fix, both caught by tests that did not exist yet

Worth recording because they are the same two mistakes this project keeps making.

`"REGRESSION_SQL"` is a **substring of** `"FAMILY_REGRESSION_SQL"`, so the first
patch windowed the family query twice and the whole-query one not at all --
exactly the shape of the `NestLoop(` / `NoNestLoop(` bug in 2.18. And the double
application left `OBSERVATION_COUNT_SQL` with two placeholders and one
parameter, which the entire 1,016-test suite passed straight over: a fake cursor
accepts whatever it is handed. The real driver did not, and said so only at
runtime, against a live database.

`test_guard_evidence_window.py` now asserts the arity directly -- that every
history query carries exactly one placeholder, and that `refresh` passes exactly
one parameter to each. A test that checks behaviour through a fake is worth
little if the fake is more permissive than the thing it stands in for.



## 2.26 A single benchmark run does not measure this system

Four consecutive runs of the same configuration reported held-out headroom of
**22.0%, 5.6%, 3.0% and -1.7%**. That looks like noise, and 2.2 gives a good
reason to expect noise -- repeated executions of the same plan on this database
differ by 18.6% on average, and the held-out set is 11 queries.

It was not noise. Running the benchmark five times in a row, changing nothing:

| run | headroom (all) | headroom (held-out) |
|---|---|---|
| 1 | +38.5% | +8.4% |
| 2 | +36.1% | +6.2% |
| 3 | +55.4% | **+46.3%** |
| 4 | +54.1% | +44.2% |
| 5 | +48.4% | +34.4% |

| | median | spread |
|---|---|---|
| headroom captured, all 24 | **+48.4%** | +36.1% .. +55.4% |
| headroom captured, held-out | **+34.4%** | +6.2% .. +46.3% |
| end-to-end faster | **+19.5%** | +13.1% .. +21.0% |

Native total itself varied by **2%** across those runs, so the workload
measurement is stable and the movement above is not the database wandering.

### It is the guard learning, and that is the design working

`regression_guard.caution_multiplier` raises the confidence bar for a query the
system has not served, and eases it back as evidence arrives -- 2.0 with no
history, 1.0 once `min_observations` served executions exist. Its docstring
says why: *"Deviating on a new query stays possible, it just has to look clearly
worth it."*

2.25 narrowed the evidence window to the current model's promotion. So a model
promoted a minute ago has **no** history under that window, every query is
unseen, and every query starts at caution 2.0 -- which doubles the required gain
and halves the tolerated regression (2.21).

Measured, at the two ends of those five runs:

    at promotion   caution 2.00 x13,  1.67 x9,  1.33 x2
    after 5 runs   caution  median 1.00,  max 1.67,  observations median 7

The gate did not change. The model did not change. The system accumulated
evidence about itself and, as designed, became willing to act on it.

### What that means for every number in this document

**A benchmark taken immediately after a promotion measures the cold-start
case.** Every earlier single-run figure in this section was taken exactly there
-- right after promoting a model or applying a gate -- which is the most
pessimistic moment in the system's life, and is why they read so low and moved
so much.

Quoting either end alone is misleading in a different direction, so both belong
together: **cold, the system captures roughly 6-8% of held-out headroom; warm,
roughly 35-46%.** The median of five runs, +34.4%, is the number to carry
forward, with the spread attached to it.

There is a real operational consequence, and it is not obvious: **promoting a
model temporarily makes the system more conservative**, because promotion resets
the evidence window that the caution multiplier reads. That is the correct
behaviour -- a new model has earned nothing yet -- but it means a deploy is
followed by a period of reduced benefit rather than an immediate step change,
and a benchmark run during that period will understate the model that was just
deployed.


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

| Join-order sampling | Usable candidates |
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

**A bind-mounted data directory on Windows was exposed to host file locks.**
Demonstrated the hard way: a host process took an exclusive lock on one index
file, a `REINDEX` intended to route around it committed and then could not
unlink the old file, and PostgreSQL panicked and would not restart -- WAL redo
kept replaying a commit that referenced a file it could not open. Nothing was
lost and `wsl --shutdown` released it, but a transient read error had been
turned into an unbootable database, and the lock recurred three times on three
different relations.

The bind mount was a deliberate choice -- "where is my data" with one visible,
copyable answer that a stray `docker compose down -v` cannot destroy -- and the
reasoning still holds. What it did not account for is that a PostgreSQL data
directory on a Windows filesystem is open to every other process on the host:
Defender's real-time scanner and the Search indexer both take handles there, and
a file open elsewhere is a file PostgreSQL cannot open.

`docker-compose.yml` uses a named volume now. It lives inside the Docker VM
where no host scanner reaches it, which makes the failure class impossible
rather than unlikely, and it avoids crossing a 9p/virtiofs boundary on every
page. What is lost is seeing the files in Explorer, and what anyone actually
wanted from that is a backup -- `pg_dump` gives a better one: portable across
versions, verifiable with `pg_restore --list`, and a fifteenth the size.

### Already present, just undocumented

**Scheduled retraining.** The list said retraining was "triggered, not
scheduled". A background scheduler has been there all along:
`AUTO_RETRAIN_SECONDS` runs `retrain_if_needed` on an interval, off the event
loop, holding a reference to the task so it cannot be garbage collected
mid-await, and cancelling it cleanly on shutdown. It is off by default because
a job that silently swaps the served model should be opted into. The limitation
was a documentation gap, not a missing feature.

### Still open

**On unseen queries it now captures 45.7% of the available headroom, and the
spread is 19.3 to 50.4.** This entry has been rewritten three times and all
three are worth keeping, because the mistakes were more instructive than the
result.

The symptom was held-out capture of **exactly 0.0%** on every run ever measured
-- eleven across three sets (2.37), three more this session -- while five of
eight held-out queries had 20-43% available. 2.37 blamed the gate for declining
opportunities and prescribed a larger workload. 2.41 ran that measurement and
concluded the opposite: with the gate fully open the model was net -2.93%, so
there was nothing behind the gate to take.

Both were measuring `greedy`. The container serves `pairwise_rank` (2.43). And
the replay could not see the safety veto at all, so it scored plans the server
refuses (2.47). Corrected, the benchmark reads:

| | measured |
|---|---|
| headroom captured | 41.3% (38.9 - 43.0) |
| **held-out captured** | **45.7% (19.3 - 50.4)** |
| curated only, no generated queries | 33.0% (20.8 - 34.2) |

Nothing in the model, the features or the action space changed to produce that.
Six corrections did, each silently wrong for months: 2.42, 2.43, 2.44, 2.45,
2.47, 2.50. The system was capable of this the whole time; what it lacked was a
measurement that agreed with what it was running.

**What is still open is how much to trust the number.** A 31-point spread across
three runs is enormous, on the same instrument 2.32 caught giving two identical
configurations medians eight points apart. The benchmark's own summary says
differences under 4.2 points should not be argued about. The honest claim is
"clearly not zero and probably around a third to a half", not "45.7%".

And the gap between the headline and the curated-only figure -- 41.3% against
33.0% -- is the selection effect, measured rather than warned about for the
first time. 27 of the 59 queries were generated and kept *because* the best plan
in the action space beat native by 8%.

**30.1% of available headroom, on queries the model has never seen.** The three
ensemble policies all land above 100% -- worse than leaving PostgreSQL alone --
and one of them was what every benchmark, every calibration and every promotion
decision in this project had measured.

So the limitation is no longer "the model does not work". It is that the project
could not tell, for its entire history, whether it worked, because the thing it
measured and the thing it ran were different. The numbers in section 2 predating
2.43 should be read as statements about `greedy`.

What is genuinely still open is the size of the remaining gap: 95.1% against an
oracle of 83.6% means two thirds of the available headroom is still unclaimed,
and `docs/ROADMAP.md` item 2 sets out why the ranker rather than the ensemble is
where that work belongs.

**Prediction error against available headroom.** Now 22,544 servable rows
across 59 queries -- fifty times the data this entry was first written about --
and the conclusion has not changed. 2.35 measured **2.51x** typical
multiplicative error on trained queries and **2.66x** on held-out ones; the
model promoted on 2026-09-04 measures **1.32x**, which is better and still the
same order as the speedups worth chasing. `app.diagnostics` puts it plainly:
*predictions are typically off by 32%, which is the same order as the speedups
worth chasing*.

2.33 established the model generalises fine and is *uniformly* inaccurate, which
points at the features and the label rather than the sample size.

**2.43 sharpens what that means rather than softening it.** Uniform inaccuracy
is survivable for a *ranker* and fatal for a *regressor used as a ranker*: a
2.5x error that is roughly constant across a query's candidates mostly cancels
in a comparison and not at all in a prediction. That is why `pairwise_rank`
lands at 95.1% of native on held-out queries while every policy reading the
ensemble's point predictions lands above 100%. The error bar has not shrunk;
the system stopped depending on it.

**The ensemble spread is not a standard deviation, and `fixed_z` assumes it
is.** `uncertainty_scale` is **3.75**: the members agree with each other that
much more than they agree with reality. Bootstrap resampling measures sampling
variability and is blind to model bias and measurement noise, both of which
dominate here (2.4.5).

This was the top fix on the list until 2.42, and it is downgraded because the
gate no longer has to believe the spread. The conformal bound is an axis of
`app.calibrate`'s sweep now, so a bound taken from the empirical quantile of the
model's own out-of-bag residuals competes with `mean + z * sigma` on measured
terms, per bundle. Over 15 held-out queries `fixed_z`'s net moved from **-4.04%
to +1.17%** depending where `min_relative_gain` was set, worst case 1.73x;
`conformal` returned **+1.10%** at every value of that knob, worst case
**1.07x**. The best single number still belongs to `fixed_z` -- by 0.07 points,
far below what this project's measurement can resolve (2.18, 2.32). The 5-point
swing is not.

Both columns of that comparison were measured under `greedy`, and re-scoring
them under the served policy reverses the verdict entirely (2.46): `fixed_z` at
z=2.0 returns **+3.10%** with a worst regression of **1 ms**, while `conformal`
returns **-11.64%** with a worst regression of **293 ms**. A constant-width
bound is too loose exactly where the ensemble is unsure, and the ranker's worst
picks are where it is unsure.

So neither bound is settled, and the durable part is that `app.calibrate` scores
both per bundle under the served policy and records which it chose. The current
answer is `fixed_z, z=2.0` on one model and 15 queries.

What remains is that `uncertainty_scale` is a standing reason to distrust
`sigma` as a probability anywhere else it is read.

**The workload is partly selected to be winnable.** 27 of the 59 queries were
generated and then kept *because* the best plan in the served action space beat
native by at least 8%. That was the right call for training coverage -- a query
with nowhere to go teaches nothing -- but it means any capture percentage over
this workload is measured on a benchmark chosen, in part, for having something
to capture.

`app.benchmark` prints the 18 correlation traps, 6 controls and 8 shape queries
as their own subtotal now, next to the headline. Those were written by hand
against a mechanism and never filtered on whether this optimizer could win them,
so the gap between the two figures *is* the selection effect, rather than
something a reader has to take on trust.

The split also carries information. On held-out queries the model does **worse**
on the generated ones -- 80% of deviations regress against 33% on the curated
ones (2.41) -- which is the opposite of what selecting for headroom would
predict, and one more argument that the ranking rather than the opportunity is
the binding constraint.

**Measurement cannot resolve the differences being argued about.** Three
compounding problems, all measured: repeated executions of the same plan vary
**18.6%** (2.18); the benchmark mutates the regression guard's state, so two runs
of an identical gate gave 15.6% and 7.2% (2.30); and two independent 3-run
measurements of an identical configuration gave medians 8 points apart (2.32).
The `compare` protocol repeats and reports spread rather than a point, and the
usable test is whether ranges overlap -- but at five repetitions it still cannot
settle a 4-point difference, and most differences worth arguing about are
smaller than that. The 2026-09-04 run's held-out figure spans **19.3 to 50.4**
across three runs, which is the problem in its most vivid form: the headline
metric's range is wider than the metric.

**This is now a constraint the tooling respects rather than only reports.**
2.48 and 2.52 are both instances of a selection procedure acting on a difference
below its own resolution -- once by maximising over a stochastic policy's best
draw, once by recommending a policy switch on 0.8 points. So `app.autotune`
draws stochastic policies five times and scores the median, and requires a
challenger to beat the *running* policy by `MIN_POLICY_SWITCH_POINTS = 2.0`
before recommending a change. Deliberately looser than the 4.2 execution can
resolve: enough to stop the obviously-noise flips without pretending replay
settles what execution cannot.

The general rule, arrived at from three directions now: **a selection procedure
has to know the resolution of its own instrument and act only on differences
larger than it.**

**The safety veto is structural, not learned.** It asks whether PostgreSQL
flagged the plan as one it was forced into -- `Disabled: true` nodes on PG18,
`disable_cost` before that -- and not whether the plan is likely to be slow.
Distrusting cost estimates is the premise of the project, so a veto that read
them would be arguing with itself; the version that did, before 2.29, vetoed
every genuine win it found, because the plans worth serving are exactly the ones
PostgreSQL costs above its own choice.

What it therefore cannot catch is a plan that is structurally ordinary and slow
anyway. That is the confidence gate's job, and 2.47 measured the two overlapping
more than expected: on the deployed model's eight held-out queries the veto
fired on three, and in all three every remaining candidate was predicted worse
than native, so the gate would have declined them regardless.

The one repair measured and rejected is falling through to the runner-up when
the veto fires. Walking the ranker's ordering until a candidate clears the gate
scored **-3.00%** against top-1-only's **+1.00%** at a loose setting, and tied
at a tight one. Candidates ranked below the top pick are worse, and letting the
gate authorise them admits marginal plans without finding better ones.

**Cold start.** Before a model exists, selection falls back to the cost
heuristic. Reasonable, but there is no bootstrapped warm-up of the kind Neo
uses.

**Single-node, single-connection measurement.** Every latency here is measured
on an otherwise-idle database. Real workloads have concurrent queries competing
for buffer cache and I/O, which changes the arithmetic on candidate execution
cost.

**Synthetic skew is still synthetic.** `data/schema.sql`'s correlations are
deliberately clean compared to real data's long tails, which is exactly why the
JOB results matter for external validity -- and why the synthetic numbers should
not be assumed to carry over.

2.55 is the first time they have been checked rather than caveated. The
synthetic workload went from 0.0% to 45.7% held-out capture and real IMDB went
from 0% to reaching the oracle exactly on its three expensive queries, in the
same week and from the same set of corrections. Two databases with nothing in
common agreeing that something changed is stronger evidence than either alone --
and it is the direct answer to the obvious objection, that six fixes to a
*measurement apparatus* might produce a better-looking number rather than a
better system. JOB is a schema nobody here designed, a workload nobody here
wrote, and it moved.

What remains true is that a clean synthetic correlation is an easy version of
the phenomenon, and that `data/schema.sql` was built to contain exactly the
trap this system exploits. A win there is a demonstration; the JOB numbers are
the evidence.

**The JOB result is answered: 0% became 99.4%, and it is concentrated.**
`docs/JOB_RESULTS.md` recorded this system capturing **none** of a 75% oracle gap
on the real 21-table IMDB benchmark, every selector returning the native plan to
the second decimal place. Everything built since had never been tested against
it, because the `job` database did not survive the upgrade.

Reloaded -- 8.7 GB, 73.0M rows -- and re-run on 27 queries with **7 held out**
against the original's 2. The held-out queries average **4009 ms** native
against the original's 3703 ms, the same expensive-query regime, which is what
makes it a comparison rather than a different measurement:

| selector | avg latency | headroom captured |
|---|---|---|
| native | 4009.4 ms | — |
| cost heuristic | 4356.6 ms | *worse than native* |
| oracle | 1463.4 ms | 100% |
| **`pairwise_rank`** | **1479.7 ms** | **+99.4%** |
| `greedy` | 2405.6 ms | +63.0% |

**Where it comes from matters as much as the number.** On the three expensive
queries -- `17e` at 10.4 s, `17d` at 8.8 s, `6f` at 8.0 s -- the ranker picks the
**oracle plan exactly**, not near it, cutting them to 0.28x, 0.28x and 0.51x of
native. Those three are 100.3% of all the saving. On the four cheap queries it
is mixed, with two regressions totalling **-82.2 ms** against 17,757 ms saved,
one of which (`5b`) is a query where no candidate beats native at all.

That weighting is appropriate for a benchmark about expensive joins and it is
not a claim the model is right about every query. Seven held-out queries is more
than two and still few, and the cost heuristic being worse than native here
flatters anything that is not cost-based.

What it does confirm, from a completely different direction than 2.43: the
ranker beats every policy reading the ensemble's point predictions, by 36 points.

**And it does not answer the original question.** The earlier run's held-out
queries averaged **3703 ms** native; these average **185 ms**. Both runs asked
for the smallest JOB queries and got different ones into the held-out quarter,
so this is a new measurement rather than a re-run, and the specific claim -- 75%
available on expensive queries, 0% captured -- has not been retested. The
collection needed to retest it is still running, because the `17*` family times
out on most of its candidates at 120 s each.

Four held-out queries is also too few to lean on: a rate over four moves in
steps of 25%.

**Fifteen-plus silent bugs found, and the rate is not falling.** Every failure
catalogued in section 2 produced a plausible number as its only symptom -- a
hint that did nothing, a veto that could not fire, a metric that could not see a
4x error, an advisor returning an empty list because an extension was never
created.

The count was "eight-plus" until 2.42-2.52 added seven more in a single session,
including the two most consequential in the project's history: the gate
calibrated under a policy the server does not run (2.43), and a replay that
could not see the safety veto and so scored plans the server refuses (2.47).
Between them they account for the entire difference between 0.0% and 45.7% of
held-out headroom captured. Neither produced an error, a warning, or an
implausible number at any point.

Three things are worth extracting from that, because "there are probably more
bugs" is not actionable on its own:

- **The productive question was "is this code reached", not "is this code
  correct".** It found `pg_stat_statements` never created (2.36), two hint
  families switched off and never re-measured (2.27), a conformal gate fitted
  on every training run and consulted on none (2.42), and a swept `gate_mode`
  the API server ignored (2.50). Every one was correct code that nothing ran.
- **The second productive question was "does this measurement agree with what
  runs".** Policy (2.43), action space (2.50), candidate pool (2.47), promotion
  yardstick (2.45), drift window (2.49). Five instances, all of the same shape:
  a number computed over a configuration nobody serves.
- **Checks now exist for the three that are checkable.** `models/gate.json`
  records the policy, the action space and the moment it was applied, and
  `app.diagnostics` compares each against reality. That is a response to the
  base rate rather than an answer to it -- but it is a response that would have
  caught 2.43 in seconds, and did catch 2.52's repeat of it.

## 2.27 The two switched-off families do not earn their place

Forced join methods and bushy trees were both built, both fixed so they bind
correctly (2.20), and both left switched off. The standing note was that turning
either on is *a measurement, not an edit*. This is the measurement.

**Method.** For each of the 24 workload queries, execute every candidate the
served action space produces and every candidate the two extra families add, and
compare the best *actually executed* latency of each set. Each candidate is run
three times and the median taken, because 2.18 measured 18.6% run-to-run
variation on this database and a single execution cannot resolve the question.
The comparison is on the best candidate available, not on what the model picks,
so a weak model cannot hide a strong family.

**Result.**

| Extra-family measurement | value |
|---|---|
| queries measured | 24 |
| queries where an extra-family candidate wins | **0** |
| best-candidate total, served space | 1620.9 ms |
| best-candidate total, plus both families | 1618.5 ms |
| improvement | **0.15%** |
| extra candidates planned and executed | 254 |

0.15% against 18.6% noise is nothing. Three queries moved at all -- 24.7 to
24.1, 116.0 to 114.5, 92.2 to 92.0 -- and none by more than the measurement can
resolve.

**They stay off.** 254 extra candidates per pass, each costing a planning round
trip, for an improvement indistinguishable from zero. And the cost is not only
planning time: 2.18 traced a gate that could not fire to the forced join-method
family being in the *training* set, where its mean ratio of 5.49 against 1.97
dragged every prediction upward.

This is a satisfying place for that thread to end. The families were worth
building -- the work of making them bind correctly is what produced the
hint-binding readback in 2.20, which then found two more real defects -- and the
honest conclusion is still that the action space is better without them.

It also answers a question 2.4 left open about where the remaining headroom
lives. It is not in a family nobody switched on. Comparing the served best
(1620.9 ms) against native totals, the *candidates* are not the constraint; the
model's ability to pick among them, and the gate's confidence in that pick, is.
Every measurement since 2.17 has pointed the same way.


## 2.28 The registry's decision measured a policy nobody runs

2.23 stopped `train` from writing over the served model, so the registry's
promotion decision finally had effect. Which made it worth asking what that
decision actually measures.

`retrain._score_bundle` reduces each model to one number -- the average latency
of the plans it would have picked over a held-out set -- and it picked those
plans with a hardcoded `policy="risk_averse"`.

| where | policy |
|---|---|
| `docker-compose.yml`, what the stack serves | **`pairwise_rank`** |
| `retrain._score_bundle`, what promotion scored | **`risk_averse`** |
| `learned.py`, the code default | `greedy` |

Three policies named in three places, and promotion deciding on the one nobody
runs.

### It had already bitten

The model promoted on 2 September was accepted for *"beating the champion by
8.3%"*. Its own recorded eval, on the served policy:

| policy | previous model | promoted model |
|---|---|---|
| `pairwise_rank` **(served)** | 0.163 | 0.162 |
| `risk_averse` (scored) | **+0.365** | **-0.142** |
| `thompson` | +0.045 | -0.104 |
| `greedy` | -0.006 | +0.002 |

Its typical multiplicative error went from 1.177 to 1.238 and its uncertainty
scale from 3.88 to 4.40 -- both worse. On the policy that serves, flat. The
number promotion read moved in the opposite direction to the number that
mattered, and the model was promoted anyway.

(The two columns come from different splits -- `train`'s eval versus
`_held_out_rows()` -- so they are not required to agree in magnitude. The policy
mismatch is independent of that, and is the part that is simply wrong.)

### The fourth instance of one pattern

`served_policy()` now reads `SELECTION_POLICY` from the same environment the
optimizer reads, and `_score_bundle` takes the policy as an argument defaulting
to it, so a deployment that changes the setting is scored correctly without
anyone remembering to update a second place. `pairwise_rank` needed handling
explicitly -- it lives on the ranker rather than the ensemble, so it is not one
of `bandit.POLICIES` and scoring it through `select_index` raises, which
`_score_bundle` would have swallowed into "could not score both models". The
promotion reason now names the policy it was reached under.

This is the fourth time the same mistake has been found in this system:

| section | the constant | what it disagreed with |
|---|---|---|
| 2.18 | training's action space | what serving offers |
| 2.21 | calibration's implicit caution of 1.0 | the 1.33-2.0 the gate applies |
| 2.22 | collection's `include_join_methods=True` | `INCLUDE_JOIN_METHODS` |
| 2.28 | promotion's `policy="risk_averse"` | `SELECTION_POLICY` |
| 2.28 | the collector CLI's `not args.no_join_methods` | `collect()`'s own default |

Each one was a value restating a decision owned somewhere else. None was wrong
when written. Every one drifted, silently, and each was found only by comparing
two things nobody had thought to compare -- which is the same lesson as 2.20,
one layer up: **where a decision has an owner, read it from the owner.** A copy
is a bug with a delay on it.


### A postscript: the fifth instance, hiding behind the fourth

2.22 gave `collect()` a `None` default for `include_join_methods` so it would
read `planner.INCLUDE_JOIN_METHODS` rather than assert an answer of its own. The
CLI underneath it still passed:

```python
include_join_methods=not args.no_join_methods
```

which is `True` whenever the flag is absent -- so every command-line run
overrode the default that had just been added to stop exactly this. The function
was fixed and its only caller was not.

Found by watching a live collection rather than by reading either file: the
active query was `HashJoin(o u) HashJoin(o u oi) HashJoin(o u oi p)`, a family
that is switched off at serving. Measured over that run: **567 of 2,501 rows --
23% -- were join-method candidates**, collected at the slowest rate in the
action space (mean ratio 5.49 against 1.97) and destined to be discarded by
`rows_for_served_action_space` before training saw them.

Two lessons, and the second is the useful one. Fixing a function and leaving its
caller is how a bug survives being fixed. And the reason this was caught at all
is that a long-running job was watched while it ran -- neither the test suite
nor a code reading had anything to say about it, because both files were
individually correct.

## 2.29 The upgrade switched off a safety veto, silently

`docs/TECHNIQUES_BACKLOG.md` argued that `Disabled: true` was the one feature
worth upgrading PostgreSQL *for*, because it turns "this candidate did nothing"
from an inference into a fact the planner states. That was right, and for a
sharper reason than the one given.

### What the veto was doing

`learned._is_unsafe` refuses a candidate the planner considers structurally
broken -- typically a forced cartesian product. It detected them by cost:

```python
DISABLE_COST_SCALE = 1e9
...
return chosen_cost >= self.DISABLE_COST_SCALE
```

That worked because PostgreSQL priced a node reached under `enable_x = off` at
`disable_cost` (~1e10) and added it to the plan total, so a forbidden plan
announced itself. Measured on 16.15: `/*+ Set(enable_seqscan off) */` on a
two-table join cost **10000003979.3**.

### What PostgreSQL 18 did

It removed the inflation. Disabled nodes are counted separately and compared
before cost, rather than being priced out of contention. Measured on 18.6, one
query, with and without three scan types banned:

| | total cost | disabled nodes |
|---|---|---|
| no hint | **8.0** | -- |
| `enable_seq/index/bitmapscan off` | **8.0** | `Seq Scan`, `Index Scan` |

Identical cost. **No cost comparison can distinguish a plan the planner was
forbidden to use from one it chose freely**, so the veto stopped being reachable
the instant the server was upgraded -- with no error, no failing test, and a
`DISABLE_COST_SCALE` constant still sitting in the file looking like it did
something.

This is the project's own recurring failure mode arriving from a new direction.
Every previous instance was a value that drifted from a decision owned elsewhere
(2.18, 2.21, 2.22, 2.28). This one is a **fact about the world** that the code
encoded and the world then changed. Nothing in the repository was edited; a
dependency was upgraded, and a safety mechanism became decorative.

### The fix, and why it is better than what it replaces

`plan_extractor.count_disabled_nodes` walks the tree and reads the flag;
`_is_unsafe` refuses on a non-zero count and keeps the cost test underneath it,
because that one is still correct on PG16 and costs nothing to leave in.

Reading the fact beats inferring it on two counts beyond simply working:

* It says **which** nodes were disabled, not merely that the total looked large.
* It cannot be fooled by a plan that is legitimately expensive. The cost test
  had a threshold, and a threshold on a quantity with no natural scale is a
  guess -- 2.4's measured example, a candidate costed 1.8x native that ran 6x
  faster, is exactly the kind of plan a cost-based veto risks catching.

### It was not only the veto

Chasing the same change through the rest of the system found three more places
that depended on the cost signal, and one of them is worse than the veto.

**A forbidden plan now looks cheaper than a correct one.** Measured on 18.6, a
four-table join:

| `Leading()` order | | total cost | disabled |
|---|---|---|---|
| `o u oi p` | connected | **6990.8** | none |
| `p o oi u` | disconnected | **20.5** | `Nested Loop` |

The unusable plan is **340x cheaper** on the planner's own estimate. That is not
a cosmetic problem:

* `_select_heuristic` picks the minimum cost, so it would take the broken plan
  every time -- and it is the fallback whenever no model is loaded and the
  tie-break for `pairwise_rank`.
* `total_cost` is a **model feature**. A training row pairing cost 20.5 with a
  plan that cannot run teaches exactly the wrong thing.
* The candidate consumes a budget slot that a real alternative would have used.

So the flag is now read in four places: `planner.plan_query` drops such plans
from the action space, `collect_data` stops recording them,
`train.rows_without_disabled_plans` removes the ones collected before either
filter existed, and `stats.COST_VS_LATENCY_SQL` keeps them out of the
cost-vs-latency chart.

That fourth one was found later, by reading the docs line by line rather than
the code: `docs/METRICS.md` asserted that disabled plans are excluded from the
chart "because PostgreSQL adds `disable_cost` (10^10)". The exclusion was a
`total_cost < 1e9` floor, and on PG18 nothing reaches it. Measured on the log:
**8,576 rows plotted, 1,157 of them (13.5%) carrying `Disabled`, and 0 caught by
the floor.** The filter had been entirely dead.

The reasoning behind it had also inverted, which is the part worth keeping. The
original argument was that disabled plans would *flatter* the correlation, being
"reliably both top-of-scale and slow". On PG18 they are bottom-of-scale and
slow, so the same rows were dragging the correlation down and making the cost
model look worse than it is. A stale comment can be wrong in the opposite
direction to the one it warns about. **Measured on the freshly collected PG18 log: 258 of 2,135 rows
-- 12% -- carried a genuinely disabled plan.** Every one would have been a label
asserting that a plan the planner refused to build is fast.

The comments were wrong too, in four places, all variations on *"Postgres prices
that at disable_cost and never picks it"*. They were true when written and are
now precisely backwards: the connectivity filter in
`generate_join_order_candidates` was described as saving planning effort, when on
PG18 it is the thing standing between a minimum-cost selector and a cartesian
product.

### The general point

An upgrade is not only a source of new capability; it is a source of **silent
capability loss**, because the thing that breaks is the assumption a piece of
code was written against rather than any line in it. The test suite caught the
`DisableIndex` change on the first run after the upgrade, because a test had
been written to fail when the world changed (2.20). Nothing caught this one --
there was no test asserting that a forbidden plan is refused, only tests that
`_is_unsafe` returns True for a cost above a constant.

A test that pins the mechanism survives the world changing. A test that pins the
number does not.


## 2.30 The benchmark changes the system it is measuring

2.26 established that a single benchmark run does not measure this system,
because run-to-run variation on this database is 18.6%. Re-benchmarking on
PostgreSQL 18 found a second, worse problem: **the benchmark mutates state the
next benchmark reads.**

Three consecutive runs:

| run | gate | guard blocks | overall | held-out |
|---|---|---|---|---|
| 1 | `z=1.0, gain=0.1` | 0 / 24 | **15.6%** | **21.6%** |
| 2 | `z=1.5, gain=0.1` (recalibrated) | 0 / 24 | -3.8% | -14.0% |
| 3 | `z=1.0, gain=0.1` -- **the same gate as run 1** | **10 / 24** | 7.2% | 7.7% |

Runs 1 and 3 differ only in when they happened. Between them, the regression
guard accumulated history from runs 1 and 2, and began blocking ten queries it
had watched regress. The guard is doing exactly its job. It also means the
measurement is not repeatable, and that **run 2 cannot be attributed to the gate
change** -- it shares its state with whatever preceded it.

This invalidates the obvious way to evaluate a gate setting: apply it, run the
benchmark, compare the number. Every run leaves the guard in a different place,
so the second measurement is taken on a different system.

### The protocol this project uses

There were three options, and the resolution is that **there are two questions,
not one**, and they need opposite treatment:

| protocol | question | guard | passes |
|---|---|---|---|
| **`compare`** | is configuration X better than Y? | **off** | **3** |
| **`deploy`** | what is the running system delivering? | on | 1 |

`compare` turns the guard off, and this costs nothing in fidelity. A guard with
no history blocks nothing anyway: it requires `min_observations` served
executions before it will act, and a single workload pass supplies one. So
"reset the guard between runs" and "disable the guard" produce the *identical*
first run -- disabling is simply honest about which is happening, and does not
pretend to a control it is not providing.

`compare` also repeats, because 2.26 measured 18.6% run-to-run variation and a
single pass cannot resolve a difference smaller than the noise. It prints the
median, the range, and the sentence that matters: *a difference smaller than
this sample's own spread should not be argued about.* Most of the differences
this project has argued about were smaller than that.

`deploy` keeps the guard on, does one pass, and its number is a point on a
trend. It is never comparable with a `compare` number, and the output says so
rather than leaving a reader to notice.

The rule that falls out: **the question travels with the number.** Every
protocol carries the question it answers, and the summary prints it above the
result, because the failure this exists to prevent is not a wrong measurement --
it is a right measurement quoted against the wrong question.

For the record, the rejected options and why:

* **Reset the guard between runs**, making each measurement independent and
  losing the thing the guard exists to provide.
* **Measure with the guard disabled** (`--no-guard` already exists), which
  measures the model and gate in isolation and overstates what production does.
* **Treat the benchmark as a sequential process** and compare only runs from the
  same starting state, which is the most faithful and the most expensive.

What is *not* honest is quoting a single number without saying which run it came
from and what the guard had already learned. Every headline figure in this
repository predating this section was measured without that qualification.

The recalibrated gate was reverted, but not because run 2 proved it worse --
run 2 proves nothing on its own. It was reverted because the incumbent has a
measurement behind it and the challenger does not, and reverting is the cheaper
mistake to undo.

### What the protocol said the first time it was used

Immediately, and it is not flattering:

```
headroom captured    -5.3%   range -13.6% to -2.0%   runs -5.3, -13.6, -2.0
held-out captured     6.7%   range  -1.7% to  7.6%   runs -1.7,  6.7,  7.6

A difference smaller than 11.7 points is inside this sample's own spread.
```

Three controlled passes on PostgreSQL 18 say the system **does not currently
beat native overall** -- median -5.3%, and every individual run negative. The
held-out median is positive at +6.7%, but its range crosses zero.

Set against the single runs quoted earlier in this same session -- 39.2% on
PG16, 15.6% on PG18 -- the gap is the whole point of the exercise. Those were
each one pass, and the spread measured here is **11.7 points**, wider than most
of the differences this project has spent effort arguing about. They were not
wrong measurements; they were unrepeated ones, and an unrepeated measurement on
this database cannot distinguish a real effect from a draw.

Two things follow, and the second is uncomfortable.

The system's honest current standing on PG18 is **around break-even overall,
with a positive but noisy held-out signal**. That is a worse claim than any
previously made in this document and it is the first one with error bars on it.

And the protocol earned its place in a single use. It was built to stop the
guard confounding comparisons, and the first thing it did was overturn the
headline figure. A measurement method that only ever confirms what you already
reported is not measuring anything.


## 2.31 The promotion metric could not see the failure that mattered

The `compare` protocol (2.30) said the system was losing on PostgreSQL 18:
median **-5.3%** captured headroom across three controlled runs. Reading the
per-query output rather than the summary showed the losses were not spread
around. One hint dominated them:

| query | native | served | model predicted |
|---|---|---|---|
| `corr_3w_brand_chain` | 137.3 ms | **160.4 ms** | 0.27x |
| `corr_3w_city_electronics` | 96.9 ms | **133.6 ms** | 0.75x |
| `corr_brand_band` | 68.8 ms | **85.7 ms** | 0.58x |
| `corr_brand_category` | 68.7 ms | **83.2 ms** | 0.53x |

All four are `Set(max_parallel_workers_per_gather 0)`. Predicting 0.27x and
measuring 1.17x is not a near miss.

Two hypotheses were wrong before the right one turned up, and both are worth
recording because they were plausible:

* **A measurement-order bias.** `collect_data` measures the baseline *once*,
  first, then each candidate `reps` times and takes the median -- so the
  denominator of every training label carries full run-to-run noise while the
  numerator has it filtered out, and the baseline runs on the coldest cache.
  Real, and measured at **1.8%** (median-of-3 over first-sample, 8 queries).
  Far too small to explain a 4x error, though worth fixing on its own.
* **A missing feature.** If the model cannot see parallelism it cannot learn
  that removing it hurts. It can: `n_gather` and `n_parallel_aware` have been
  in the vector all along.

### What it actually was

The deployed model was trained on PostgreSQL 16. Asked about that exact
query and hint:

| | predicts | spread |
|---|---|---|
| champion, PG16-trained, **deployed** | **0.291x** | 0.397 |
| challenger, PG18-trained | **1.273x** | 0.134 |
| the training data's own answer (147.4 / 130.0 ms, n=20) | **1.134x** | |

The challenger is nearly right and confident. The champion is wrong by a factor
of four and, with a spread of 0.397, uncertain enough that a wider gate would
have caught it -- but the point estimate cleared the bar.

**And the registry had already compared these two models and scored them 1.0%
apart**, below its 2% promotion bar, so it kept the champion.

### Why the comparison missed it

`retrain._score_bundle` reduces a model to the average latency of the plans it
would have picked across a held-out set. On most queries both models pick the
same plan, so the average barely moves. The difference lives entirely in the
handful of queries where they diverge -- and on those, the champion picks a plan
that runs 17% slower while predicting a 73% speedup.

An average over queries answers "how good is this model on a typical query".
Promotion needs to answer "is this model ever catastrophically wrong", and those
are different questions. A metric that cannot distinguish a model that is
uniformly mediocre from one that is usually fine and occasionally disastrous is
not measuring the thing the safety gate exists to protect against.

This is the fourth time in this document that an aggregate has hidden the case
that mattered: 2.2 (a mean over queries hiding a regression), 2.19 (a regression
*rate* hiding that every regression was 1%), 2.30 (a single run hiding an 11.7
point spread), and now a mean over held-out queries hiding a 4x prediction
error. **The aggregate is where the interesting case goes to die**, and the
consistent fix has been to report the distribution -- worst case, spread, or the
per-item breakdown -- beside it.


## 2.32 The protocol measured itself, and failed

2.30 built a `compare` protocol -- guard off, three repetitions, report the
spread -- because a single benchmark run could not resolve anything. Used in
anger on the champion/challenger question, it worked: the PostgreSQL 16-trained
champion measured **-5.3%** captured headroom (range -13.6 to -2.0) against the
PG18-trained challenger's **+8.0%** (range 7.4 to 10.8). Non-overlapping ranges,
a system that had gone from net-losing to net-winning, and the promotion in 2.31
was justified.

Then the same three-run measurement was repeated on the same model:

| measurement | median | range |
|---|---|---|
| set A | **8.0%** | 7.4 - 10.8 |
| set B | **16.0%** | 8.2 - 16.6 |

Two controlled measurements of an **identical configuration**, differing by 8
points. Nothing changed between them but time.

The protocol had already said so. Its summary prints:

> A difference smaller than 8.4 points is inside this sample's own spread and
> should not be argued about.

That line was describing the sample it had just taken. It turned out to be
describing the protocol as well.

### What survives, and what does not

Pooling all six runs of the challenger against the champion's three:

    champion    -13.6  -5.3  -2.0
    challenger    7.4   8.0   8.2  10.8  16.0  16.6

The ranges do not overlap. **That** is the claim the data supports: the
challenger is better. What it does not support is *by how much* -- "13.3 points
better", which the first comparison seemed to license, was reading a median
difference smaller than the spread either sample would show on a re-run.

The default repetition count is now five. That is not enough either, and saying
so is the point: the usable test here was never "is the median higher", it was
"do the ranges overlap". A protocol that reports a spread and then compares
medians across it has built the instrument and ignored the reading.

### The held-out column, and why it is not measuring what it looks like

Held-out capture went the other way: **+6.7%** for the champion against **-4.8%**
and **-2.2%** for the challenger, with spreads of 19.8 and 14.4 points. The
obvious reading is that the new model generalises worse. Breaking the column
down per query says otherwise:

| held-out query | native | served | what happened |
|---|---|---|---|
| `corr_4w_cancelled_premium` | 71.0 ms | **67.8 ms** | acted, and won |
| `corr_city_country` | 60.5 ms | 60.5 ms | vetoed |
| `corr_brand_band` | 75.0 ms | 75.0 ms | vetoed |
| `corr_5w_supplier_chain` | 93.7 ms | 93.7 ms | vetoed |
| `ctl_3w_recent` | 125.6 ms | 125.6 ms | vetoed |
| `ctl_supplier_rating` | 5.8 ms | 5.8 ms | vetoed |

**The model acts on one held-out query in six**, and on that one it is right.
The other five are refused by the gate, which is what the caution multiplier is
for: a query with no served history gets a stricter bar (2.21).

So the held-out percentage is a ratio whose numerator is contributed by one or
two queries and whose denominator is the headroom of all six. A single win or
loss moves it by ten points or more. That is not a measurement of
generalisation; it is a measurement of whichever query the gate happened to
allow, and it explains the 19.8-point spread far better than any story about
the model.

Which leaves the real question unanswered rather than answered badly. Whether
this model generalises cannot be read off a metric dominated by five abstentions
-- it needs either many more held-out queries, or a measurement of prediction
*accuracy* on the held-out set independent of whether the gate chose to act.
The second is cheap and does not exist yet.

The overall improvement is real. The generalisation question is open, and the
instrument currently pointed at it cannot resolve it.


## 2.33 It generalises. That was never the problem.

2.32 left the generalisation question open and said the instrument pointed at it
could not resolve it. `app.generalisation` is the instrument that can: for every
logged candidate with a real latency, how far is the predicted ratio from the
actual one, split by whether the query was in the training set. It has an answer
whether or not the gate ever fires, which the benchmark's held-out column does
not.

Building it took three corrections, each the same class of mistake this
document keeps recording.

* **Scoring rows the model was never taught.** `_load_rows()` returns everything,
  including the 567 forced join-method and 294 disabled-plan rows that `train`
  deliberately drops. Measuring the model on families it will never be asked
  about is the train/serve skew of 2.18 -- committed while building the
  instrument for it.
* **Scoring raw executions against a model fitted on medians.** That charges it
  for the 18.6% run-to-run variation it was never asked to predict.
* **Writing a number before measuring it.** The comment explaining the second
  fix claimed aggregation would bring typical error to "1.4x". It was a
  prediction, typed as though it were a result. Aggregating actually moved it
  from 2.21x to **2.93x** -- collapsing 4,764 rows to 432 strips the weighting
  that repeated easy candidates had been getting. The comment now says what
  happened.

### The answer

| | typical error | p90 | n |
|---|---|---|---|
| trained on (18 queries) | **2.93x** | 4.18x | 432 |
| held out (6 queries) | **3.25x** | 4.12x | 137 |

Ratio **1.11**. The model predicts unseen queries about as well as the ones it
learned from. **It generalises.**

It also predicts both of them badly. Typically a factor of **three** out, on a
target that lives between roughly 0.3x and 3x. That single number explains what
several earlier sections found the hard way:

* 2.21's uncertainty bands are wide because the model *is* inaccurate -- 1.43x,
  1.56x, 1.58x pessimistic bounds are an honest ensemble reporting honest
  disagreement, not a miscalibration to be tuned away.
* The gate vetoes five held-out queries in six because a model three times out
  cannot clear a bar that asks for confidence.
* 2.31's champion predicting 0.291x where the truth was 1.134x is not an
  outlier. It is this error distribution, sampled once, on the candidate that
  happened to matter.

### What this changes

Every previous section treated "why is it never confident" as a gate problem,
a calibration problem, a train/serve skew problem, or a promotion-metric
problem. Each of those was real and each fix was worth making -- the system went
from net-losing to net-winning on the strength of them. None of them was the
main thing.

**The model is not accurate enough, and it is uniformly not accurate enough.**
That is a better problem to have than memorisation, because it points somewhere
specific: more data, better features, or a different target -- not more
threshold tuning. The thresholds are now honest, and what they honestly report
is a model that does not know enough.


## 2.34 A safety gate that blocked the fix for the thing it was guarding

Three hooks were added to this repo: ruff on every backend edit, the unit suite
in the background, and a gate refusing `git commit` while either is failing. The
third one deadlocked the session within the hour.

It is a `PreToolUse` hook on `Bash`, filtered by `"if": "Bash(git commit *)"` so
it would not spawn on unrelated commands. The filter did not hold. Extending the
workload broke one test, and the very next Bash call -- a `python - <<EOF` that
would have *fixed* that test -- was refused:

```
PreToolUse:Bash hook error: Refusing to commit: the backend unit suite is failing.
```

Exit 2 on a `PreToolUse` hook blocks the tool call outright, so the edit never
ran. The state was: the suite is red, therefore every shell command is blocked,
therefore the suite stays red. A gate guarding against a broken commit had made
the break unfixable.

### What the mistake actually was

Not the gate, and not the exit code. It was trusting the matcher.

A hook that can *refuse work* has to be certain it is refusing the right work,
and certainty there means reading the command rather than assuming something
upstream filtered it. The script now parses the `PreToolUse` payload and exits 0
unless the command genuinely contains `git commit` -- belt and braces where the
braces turned out not to be attached.

That is the same lesson as 2.20, arriving from the opposite direction. There,
the check inferred a fact (`the plan changed, so the hint bound`) that the
system could have been asked for directly. Here the check inferred a fact (`this
hook ran, so the command must be a commit`) that the payload states outright.
**A guard that reasons about its own trigger from context rather than from
evidence will eventually fire on the wrong thing** -- and a guard that fires on
the wrong thing while holding a veto is worse than no guard.

### The general shape

Two properties made this bad rather than annoying:

* **It failed closed on everything, not on its subject.** A gate scoped to
  commits that instead scopes to all shell access has a blast radius of the
  whole session.
* **Its failure mode removed the means of repair.** Any mechanism that blocks
  work when something is broken should be checked against the case where the
  broken thing is the mechanism's own precondition.

Both are worth a second look before adding any future hook that can return
exit 2.


## 2.35 The missing data was shapes, not rows

2.33 measured the model at **2.93x out on trained queries and 3.25x on unseen
ones** and concluded it generalises and is uniformly inaccurate -- so the fix was
not regularisation, and more rows of the same thing had already been tried.

What had not been tried was covering the *plan space* rather than the row count.
All 24 workload queries were the same shape: a select over joins with filters.
None aggregated, sorted, de-duplicated or used a subquery, so no training plan
had ever contained a `HashAggregate`, a `WindowAgg`, a `Sort` above a join, or a
semi-join. The feature vector counts `n_aggregate`, `n_sort` and `n_materialize`,
and those columns were near-zero in every row the model had ever seen. **A
feature that never varies teaches nothing, and a plan shape absent from training
is one the model is guessing about at serving time.**

Eight queries were added covering aggregation, DISTINCT, top-N, a window
function, and semi- and anti-joins. Each was measured before being included --
native latency against the best plan the served action space actually reaches,
three executions, median -- and two candidates were **rejected** on that
evidence: a top-N showing 39% headroom on a 1.7 ms query (a large percentage of
nothing) and a five-way join that linked `suppliers` to `users` on `country`,
which ran 1.77 s at 0% headroom because it is a near-cartesian join with no key
path rather than a hard query.

### What it bought

| | 24 queries | 32 queries, with shapes |
|---|---|---|
| prediction error, trained queries | 2.93x | **2.51x** |
| prediction error, unseen queries | 3.25x | **2.66x** |
| held-out / trained ratio | 1.11 | **1.06** |
| `uncertainty_scale` | 3.88 | **3.19** |

An 18% reduction in error on queries the model has never seen, from eight
queries -- a third more workload, not a different model, a different loss, or
more data of the same kind. The ensemble's self-agreement also moved closer to
its agreement with reality (3.88 to 3.19), which is the gap 2.4.5 identified as
the thing to fix before the deviation rate can be raised safely.

### And what the promotion metric caught on the way

The champion/challenger comparison promoted the new model on a 25.2% mean
improvement. The number worth reading is the other one:

| | champion | challenger |
|---|---|---|
| mean latency over held-out queries | 295.8 ms | 221.2 ms |
| **worst pick, as a ratio of the best available plan** | **228.2x** | **1.36x** |

The incumbent's worst single pick was **228 times worse than the best plan
available for that query**. That is precisely the failure 2.31 added
`worst_ratio_vs_best` to see, and precisely the failure a mean cannot: 228x on
one query out of eight moves a mean by a few percent and moves nothing else.

### The honest limit

2.51x and 2.66x are better and still large. The model is roughly two and a half
times out on a typical prediction, against available headroom that is often
smaller than that. Shape coverage was a real gain and not a solution, and the
next thing to try is more of the same -- the workload is 32 queries against a
six-table schema, which is small in the way that matters.


## 2.36 An init script is only as good as the last time someone started from nothing

Sweeping the repository for anything still assuming PostgreSQL 16 turned up
three functional assumptions rather than the expected pile of stale comments.
Two survived contact with the new server: `SHOW lc_collate` still errors on 18,
so `index_advice.py`'s workaround is still right, and `last_idx_scan` is still
there. The third did not.

**`pg_stat_statements` was not installed at all.**

`docker-compose.yml` puts it in `shared_preload_libraries`, so the collector was
running and accumulating statistics. `postgres/init/01_extensions.sql` creates
`pg_hint_plan` and `hypopg` -- and never created `pg_stat_statements`. Preloading
starts the collector; `CREATE EXTENSION` registers the view that reads it. With
only the first, the statistics pile up somewhere nothing can look.

This is 2.0's bug from the opposite side. There, `CREATE EXTENSION` had been run
and the library was *not* preloaded, so every hint became a comment. Here the
library was preloaded and the extension was *not* created, so an entire advisor
returned nothing. Both are the same gap between "the library is loaded" and "the
SQL objects exist", and both are silent.

### Why it took a rebuild to notice

The PostgreSQL 16 cluster had the extension created by hand at some point, long
enough ago that nobody remembered. Every run since had read a working view. The
18 upgrade rebuilt the cluster from the init scripts, and the omission surfaced
on the first query -- **an init script is only as good as the last time someone
started from nothing**, and this one had not been exercised end to end in a very
long time.

Fixed in `01_extensions.sql`, and the same mismatch was waiting in CI:
`.github/workflows/ci.yml` preloaded only `pg_hint_plan`, so adding the
`CREATE EXTENSION` would have produced a green build and a broken advisor. The
two preload lists are now the same list.

### What it was worth

On the live database, immediately: **34 query-statistics findings and 4
observability findings** where there had been none. Nine advisor sections and 49
findings in total, against a section that had been silently returning an empty
list.

A fresh cluster now comes up complete -- PostgreSQL 18.6, pg_hint_plan 1.8.1,
hypopg 1.4.3, pg_stat_statements 1.12, verified by building one from scratch and
running the integration suite against it (18 passed).

One thing that did *not* need fixing, which is worth recording because the
temptation was there: `query_stats.py` reads a column renamed between 16 and 17
(`blk_read_time` to `shared_blk_read_time`) and already probed
`information_schema` for whichever exists rather than assuming. Written against
1.10, it worked unchanged on 1.12. **Code that asks the server what it has
outlives the version it was written for**; only its comment needed updating.


## 2.37 Zero is not the same as nothing to take

Eleven benchmark runs across three independent measurement sets, on the model
trained with shape coverage (2.35), reported the same held-out figure every
time:

| run set | headroom captured | held-out captured |
|---|---|---|
| 3 runs | 4.3% (range 0.6-14.3) | **0.0%** |
| **5 runs** | 5.9% (range 3.6-7.7) | **0.0%** |
| 3 runs | 4.1% (range 2.5-5.2) | **0.0%** |

An identical 0.0% eleven times is not noise. Something is deterministic.

Two explanations fit, and they call for opposite responses. Either the held-out
queries have no headroom -- in which case capturing none of it is correct and
the number is uninteresting -- or they have headroom and the gate declines it.

That is answerable from the log without executing anything: for each held-out
query, the median native latency against the median of the best candidate the
served action space actually reaches.

| held-out query | native | best available | headroom |
|---|---|---|---|
| `corr_6w_everything` | 159.9 ms | 90.5 ms | **43.4%** |
| `corr_4w_full_correlated` | 154.3 ms | 100.6 ms | **34.8%** |
| `shape_anti_join_unsold` | 81.7 ms | 59.9 ms | **26.7%** |
| `corr_4w_cancelled_premium` | 126.8 ms | 93.2 ms | **26.5%** |
| `corr_city_country` | 60.5 ms | 47.6 ms | **21.4%** |
| `corr_5w_supplier_chain` | 162.2 ms | 147.3 ms | 9.2% |
| `shape_distinct_brands` | 189.1 ms | 179.4 ms | 5.1% |
| `ctl_country_only` | 40.9 ms | 39.1 ms | 4.3% |

**Five of eight have more than 20% available. The system takes none of it.**

So the 0.0% is the gate, in full. Not a shortage of opportunity, not a model
that cannot rank -- 2.35 measured the same model at 2.66x prediction error on
these very queries, and 2.33 established that its ranking survives on unseen
queries even when its level does not.

### How the gate got here

The recalibration that produced this setting ran on **8 held-out queries** and
swept a grid whose operating points differed by 0.4 percentage points of net
gain. It chose `min_relative_gain = 0.10` -- a demand for a predicted 10% win --
because that point showed 0% regressions and +0.1% net, and every looser point
showed a regression rate it could not distinguish from noise on 8 queries.

A calibration asked to choose between options separated by less than its own
resolution will pick the one that does nothing, because doing nothing is the
only option whose downside it can bound. That is not a bug in the sweep; it is
what a sweep does when the evidence runs out, and 2.19 rewrote its refusal
message for a version of exactly this. The failure is upstream: **8 queries
cannot calibrate a threshold whose effect is worth 20% on five of them.**

### What this changes about the project's own account of itself

The README has said for a long time that the optimizer "optimizes about 44% of
queries, and that is deliberate", with the deviation rate presented as a
measured trade-off. On this workload and this gate the held-out deviation rate
is **zero**, and the trade-off being made is not the one described: it is
declining a measured 43% because a sweep over 8 queries could not prove that
taking it was safe.

That is a defensible thing for a safety-first system to do once. It is not
defensible to report it as a capture rate without saying that the capture rate
is a property of the calibration set size rather than of the model.

### Loosening it, measured

The obvious response is to lower the bar to the `0.02` an earlier, larger
calibration had picked, and run the same protocol against it:

| | strict, `gain=0.10` | loose, `gain=0.02` |
|---|---|---|
| headroom captured | 4.1 - 5.9% across 11 runs | 4.1% (range 3.6 - 7.4) |
| **held-out captured** | **0.0%, every run** | **2.1%** (range -5.7 to 6.5) |

The deterministic zero is gone, which proves the gate is now acting -- one run
came back at **-5.7%**, and a setting that can lose is a setting that is doing
something. But the median gain of 2.1% sits inside a 12.2-point spread, and
overall capture did not move at all.

**Neither setting is demonstrably better than the other**, by the rule 2.32
established: a difference smaller than the sample's own spread is not a
difference. So the calibrated setting was restored, not because it won, but
because the override did not, and reverting an unproven change is the cheaper
mistake.

That is the second time in this document a controlled A/B has come back
"unresolved" rather than wrong (2.30 was the first), and the response is the
same both times: the instrument is not sensitive enough, and making it sensitive
enough means more queries, not more runs. Eleven runs already agree with each
other to within a few points. Eight queries do not agree about anything.


## 2.38 The C++ port works, is exactly right, and does not matter

Built because it was asked for, after a profile that said not to. The honest
place to start is that profile, because it was also wrong -- in the direction
that made the case *weaker*.

`docs/TECHNIQUES_BACKLOG.md` and the project report had recorded feature
extraction at **2.43 ms** per 18 candidates. Re-measuring found **0.816 ms**:
three times smaller than documented. Splitting it further, the recursive tree
walk is 45% of that and the rest is per-table dictionary manipulation, where a
compiled language gains almost nothing because it pays the same `PyDict`
lookups.

So the target was the tree walk, and only the tree walk.

### The result

`native/plan_tree_native.cpp` is a CPython C-API port of
`plan_tree._encode_plan_tree_py`. Measured on **400 real plans** from the
execution log, median of 25 passes:

| | per plan | per query (18 candidates) |
|---|---|---|
| Python | 0.0188 ms | 0.338 ms |
| C++ | **0.0082 ms** | 0.148 ms |
| | **2.29x** | **saves 0.191 ms** |

0.191 ms, against an optimizer overhead of roughly 26 ms on a query that takes
about 110 ms. **0.7% of the overhead, under 0.2% end to end.** The profile was
right: this is not where the time goes.

### What was actually hard, and it was not the speed

Correctness. A feature vector that differs between the compiled and interpreted
paths is a silent divergence between what the model trained on and what it is
served -- the failure class this document catalogues seven instances of, every
one of which produced plausible numbers while being wrong.

So the bar was **exact equality**, not `approx`: both implementations perform the
same operations in the same order on the same doubles, so there is no numerical
reason to differ, and a difference would be a bug rather than float noise.
`tests/test_plan_tree_native.py` is 30 tests, and the one that matters compares
against every plan in the execution log.

Three specific things had to be reproduced rather than approximated, and each
would have passed a casual eye:

* `float(node.get(k) or 0)` collapses **missing, `None` and `0`** to the same
  value. An implementation treating "missing" as "skip" changes every sum.
* `join_rows_reduction` is guarded by `if largest else 1.0` -- a *falsiness*
  test, so a join estimated at exactly zero rows yields 1.0 rather than
  `(root+1)/1`.
* `cost_concentration` excludes the root node, whose `Total Cost` is cumulative.
  Including it makes the feature 1.0 for every plan ever.

The build also had to be `-O2`, not `-O3 -ffast-math`: fast-math licenses
reassociation of the sums, which would make results differ from Python in the
last bits. The one optimisation flag that would plausibly help is the one that
would break the only property that matters.

### Disposition

Kept, and optional in both directions. `plan_tree.py` imports the extension if
it has been built and runs the Python if it has not, so a checkout with no
compiler is a working checkout, and the differential test skips itself rather
than failing. Nobody has to build it; anyone who does gets a 2.29x faster
encoder and an assurance that it computes the same thing.

That is the right disposition for a change that is correct, measurable, and
worth 0.2%. Shipping it as mandatory would trade a build dependency for nothing.


## 2.39 The convenience layer cost more than the language

2.38 finished the C++ port and reported it honestly: correct, 2.29x on its
target, worth 0.19 ms of a 27 ms overhead. The question that produced it -- "what
if the whole codebase were C++?" -- deserved a measurement rather than an
opinion, so the serving path was profiled by component:

| | time | share |
|---|---|---|
| PostgreSQL `EXPLAIN` round trips | 16.00 ms | 58% |
| feature extraction (Python) | 3.08 ms | 11% |
| model inference (LightGBM, already C) | **8.40 ms** | **31%** |

**88.8% was already running in C**, so a rewrite of all 19,450 lines could
address 11% of the overhead -- under 3% of a 110 ms query. That is the answer to
the question as asked.

The interesting part was the 31%. An 8-member ensemble over 19 candidates taking
8.4 ms is a lot for a small gradient-boosted model, so it was worth asking where
the time went:

| candidates | inference |
|---|---|
| 1 | 7.54 ms |
| 19 | 8.27 ms |
| 50 | 9.23 ms |

Predicting fifty candidates cost barely more than predicting one. The cost was
**fixed**, not per-candidate -- which meant it was not computation at all.

### Where it was

    LGBMRegressor.predict(list of lists)   1.211 ms
    LGBMRegressor.predict(numpy array)     0.797 ms
    booster_.predict(numpy array)          0.092 ms

The scikit-learn wrapper spends roughly **0.7 ms per call** on input validation
and DataFrame handling before reaching the model, and `predict_all` called it
once per member -- eight times per query, on data the wrapper had to convert
from a list of lists every time.

Going straight to the booster, on a float64 array built once:

**`predict_mean_std` on 19 candidates: 8.27 ms to 0.786 ms. A 10.5x speedup.**

Against the earlier uncontended baseline that is 27.5 ms of planning overhead
down to about 20 ms -- a **26% cut**. (Stated as a projection, not a fresh
measurement: the re-run happened while a 59-query collection was using the
database, which pushed the `EXPLAIN` component from 16.0 ms to 28.2 ms. The
inference figure is pure CPU and unaffected; the round-trip figure from that run
is not comparable and is not used here.)

### The comparison that matters

| change | speedup | saved |
|---|---|---|
| C++ port of the feature encoder | 2.29x | 0.19 ms |
| **not calling a wrapper eight times** | **10.5x** | **~7.0 ms** |

**Thirty-seven times more time was recovered by deleting a convenience layer
than by changing language**, and the second one is a five-line diff in Python.

It is bit-identical, and that was checked before it was believed: across 600
real feature vectors from the execution log, all 8 members produced exactly
equal output from both paths, worst absolute difference **0.0**. A ranker whose
scores shift slightly is a different ranker.

### What this says about the original question

"Rewrite it in C++" and "stop paying for something you are not using" are both
performance work, and only one of them was worth doing here. The profile is what
distinguished them, and the profile was cheap -- three timers and twenty
minutes.

It also shows why 2.24's answer was right for the wrong reason. It concluded
C++ was not warranted because PostgreSQL dominates the overhead, which is true.
It did not notice that a third of the remaining time was a Python library
validating its inputs eight times, because it measured *language boundaries*
rather than *calls*. The right granularity for a profile is the one where the
waste is, and you find that by looking rather than by reasoning about it.


## 2.40 More data, better predictions, worse picks -- and one query explaining all of it

The workload went from 32 queries to 59 (2.35 and the `gen_` group), collection
produced 20,529 usable rows against 777 before, and the retrained model came out
with the **best prediction accuracy of any version in the registry**:

| version | held-out queries | rows | typical multiplicative error |
|---|---|---|---|
| `20260903T163742Z` (deployed) | 8 | 777 | 1.311 |
| `20260904T082635Z` (challenger) | **15** | **1437** | **1.254** |

Promotion rejected it: mean latency of the plans it picks was **25% worse**, and
its worst single pick was **3.36x** the best available plan against the
champion's 2.73x. The worst-case rule from 2.31 fired for the second time.

That is a contradiction worth resolving rather than accepting. A model cannot
predict better and pick worse unless the errors it still makes are concentrated
somewhere expensive. Scoring both models per held-out query:

| query | champion | challenger | best available |
|---|---|---|---|
| `shape_6w_supplier_reach` | 606.5 ms | **1270.5 ms** | 378.3 ms |
| `shape_distinct_brands` | 197.5 ms | **115.9 ms** | 89.9 ms |
| `shape_range_recent` | 70.8 ms | **57.6 ms** | 25.9 ms |
| ... 12 others | | roughly tied | |
| **total** | **2712.0 ms** | **3394.2 ms** | |

**One query is 664 ms of the 682 ms difference.** Remove it and the two models
are within 0.9% of each other -- 2105.5 ms against 2123.7 ms -- with the
challenger winning outright on two queries and losing on none by more than 30%.

### What that says

The rejection is correct: a 2.1x loss on a single query is exactly what the
worst-case rule exists to stop, and the mean would have hidden it in the other
direction this time. But the *diagnosis* is narrow, and narrow diagnoses are
actionable in a way that "the model got worse" is not.

`shape_6w_supplier_reach` is a six-way join under a grouped aggregate -- the
most complex query in the workload. There are **three** six-table queries among
fifty-nine, and the twenty-seven queries added in this round are almost all two
to five tables. The training distribution moved toward narrow joins, and the
model followed it.

This is the cost of the workload expansion showing up where the limitations list
predicted it would: adding queries improves coverage on average and can degrade
it exactly where coverage was already thinnest. It is also direct evidence for
the roadmap's per-join-width models, or at minimum for stratifying the training
sample by width rather than letting the generator's preferences set it.

### And a structural problem it exposes

`app.calibrate` reads the *deployed* model's held-out list. The 15-query model
was rejected, so the gate is still being calibrated on the champion's **8**
queries -- which is the constraint 2.37 identified and this whole round of work
set out to remove. The fix arrived and could not be adopted, because the thing
that decides adoption measures a different property than the thing that needed
the fix.

Two properties are being conflated: *is this model a better predictor* and *does
this model come with a better calibration set*. They are not the same question,
and promotion currently answers only the first.


## 2.41 The gate was never the bottleneck

2.40 ended on a structural complaint: `app.calibrate` read the *deployed*
model's held-out list, so a rejected challenger's better calibration set could
not be used. That is a real coupling and it is now removed -- `--bundle` points
the sweep at any bundle, and `apply_gate` records `calibrated_against_bundle`
so a gate fitted to a challenger is not silently mistaken for one fitted to the
champion. A gate setting is a property of one model's error distribution; a
gate file that does not name its model is ambiguous the moment a second model
exists.

The interesting part is what the unblocked measurement said.

### The measurement 2.37 was waiting for

2.37 found held-out capture of exactly 0.0% across eleven runs, established
that five of eight held-out queries had 20-43% available, and concluded the
gate was refusing opportunities it should have taken -- calibrated on 8 queries
over a grid finer than its own resolution. The prescription was a larger
workload. The workload is now 59 queries with 15 held out, and the same replay
sweep can now be run against that model without promoting it.

Both models, same log, same policy, `greedy`:

| | z | gain | deviates | of those regress | worst | net |
|---|---|---|---|---|---|---|
| **deployed**, 8 held out | 0.0 | 0.00 | 62% | 20% | 1.31x | **+4.68%** |
| | 0.5 | 0.10 | 12% | 0% | 1.00x | +3.75% |
| | 1.0 | 0.20 | 0% | -- | 1.00x | 0.00% |
| **challenger**, 15 held out | 0.0 | 0.00 | 93% | **50%** | **1.73x** | **-2.93%** |
| | 0.5 | 0.10 | 33% | 40% | 1.73x | -4.04% |
| | 1.0 | 0.20 | 7% | 0% | 1.00x | +1.17% |

The top-left number is the ceiling: what the model is worth with no gate at
all, following every pick. On 8 held-out queries that ceiling is **+4.68%**, and
a gate capturing none of it is leaving something real behind -- which is what
2.37 argued.

On 15 held-out queries the ceiling is **-2.93%**. Half of the deviations are
regressions and the worst is 1.73x. There is nothing behind the gate to take.

### So the diagnosis was wrong, and in an instructive direction

2.37 asked whether the held-out queries had headroom, found that they did, and
concluded the gate was the problem. Both halves were right and the inference
was not: the headroom was measured against the *oracle* -- the best plan in the
action space -- and the gate does not have access to the oracle. It has access
to the model's ranking. Headroom exists; the model cannot find it. Confirming
that the money is on the table says nothing about whether this system knows
which pocket it is in.

The 0.0% was the gate declining to act on a ranking that, followed everywhere,
loses. Widening it would have converted a null result into a negative one.

The split by group says the same thing twice:

| held-out subset | n | deviates | of those regress | net at z=0, gain=0 |
|---|---|---|---|---|
| curated (`corr`/`ctl`/`shape`) | 9 | 100% | 33% | **-1.61%** |
| generated (`gen_*`) | 6 | 83% | **80%** | **-6.90%** |

Worse on the generated queries, and those were the ones *selected for having
headroom*. A query kept because the action space contained a 15% win is a query
where the model had every chance and took the wrong branch four times in five.

### What this changes

The gate is doing its job. Every conservative setting scores 0.00% net because
it declines everything, and 0.00% is better than every wide setting on this
model. A safety gate whose refusals are all correct is not misconfigured.

The bottleneck is the ranking on unseen queries, which relocates the work:

- **2.35's conclusion holds and sharpens.** It found the model *uniformly*
  inaccurate rather than overfitted -- 2.51x typical multiplicative error on
  trained queries, 2.66x on held-out. That is not a gap that more rows close,
  and 20,529 rows across 59 queries have not closed it. The features and the
  label are the suspects.
- **`uncertainty_scale = 3.75` is now the top item, not a nice-to-have.** The
  only positive setting in the whole sweep is `z=1.0, gain=0.20`, which acts on
  1 query in 15. The gate's risk term reads an ensemble spread known to be
  mis-scaled by nearly 4x (2.4.5). Quantile regression replaces a number the
  gate is currently forced to over-trust.
- **Per-join-width models get a second, independent argument.** 2.40 reached
  them from one 6-table query carrying 664 ms of a 682 ms promotion gap. This
  reaches them from held-out ranking failing across the board.

### The uncomfortable reading, stated plainly

On queries it has not seen, this optimizer's plan choices are currently worse
than PostgreSQL's, and the only reason the deployed system does not lose time
is that a correctly-calibrated gate refuses to use them. The honest description
of the current state is *a safety mechanism that works, wrapped around a model
that does not yet*.

That is not the result this round of work was aiming for. It is a better
position than the previous one, where the same fact was hidden behind a 0.0%
attributed to the wrong cause -- and it names the next problem specifically
enough to attack.

> **This conclusion is wrong, and 2.43 is why.** Every number above was measured
> under `greedy`. The container serves `pairwise_rank`, which lands at 95.1% of
> native on the same held-out queries where `greedy` lands at 104.1% -- opposite
> sides of "is this worth running". The section is kept unedited because the
> reasoning in it is sound and nothing inside it could have revealed the error:
> it took asking a different question, *which policy does the deployment
> actually run*. That is the same question that found `pg_stat_statements` never
> created (2.36) and the conformal gate never used (2.42).

## 2.42 A method that was implemented, fitted, documented -- and never measured

2.41 ended by relocating the problem from the gate to the ranking. Before
accepting that, one thing was worth checking: the gate has two possible forms,
and only one of them had ever been used.

`optimizer/conformal.py` implements split conformal prediction. `app.train`
fits it on every run -- the challenger's gate has `n_calibration=1037` and
`q=0.2064` in log space, a 90% upper bound of **1.229x** on the point
prediction. `README` documents `GATE_MODE=conformal`. `app.settings` exposes
it in the UI. Everything is wired.

`LearnedOptimizer.__init__` read `os.getenv("GATE_MODE", "fixed_z")`, with a
comment explaining that a change of method should be settled by measurement
rather than by editing a default. That comment is correct. It is also the
reason nothing was measured: it justified the default without scheduling the
measurement, and a method reachable only through an environment variable that
nobody exports is not deployed, it is shelved.

So: score both bounds on the same held-out queries, same log, same policy.

| held-out set | bound | min_gain | deviates | of those regress | worst | net |
|---|---|---|---|---|---|---|
| challenger, 15 | `fixed_z` | 0.00 | 93% | 50% | 1.73x | **-2.93%** |
| | `fixed_z` | 0.05 | 60% | 33% | 1.73x | -1.21% |
| | `fixed_z` | 0.10 | 33% | 40% | 1.73x | **-4.04%** |
| | `fixed_z` | 0.20 | 7% | 0% | 1.00x | +1.17% |
| | `conformal` | 0.00 | 20% | 33% | **1.07x** | **+1.10%** |
| | `conformal` | 0.05 | 20% | 33% | 1.07x | +1.10% |
| | `conformal` | 0.10 | 20% | 33% | 1.07x | +1.10% |
| | `conformal` | 0.20 | 7% | 0% | 1.00x | +1.17% |

Read the two blocks as columns rather than rows. `fixed_z` scores -2.93%,
-1.21%, -4.04%, +1.17% as `min_relative_gain` moves across its plausible
range. `conformal` scores +1.10%, +1.10%, +1.10%, +1.17% across the same range.

### The interesting property is not the number, it is the flatness

`fixed_z`'s best setting beats conformal's: +1.17% against +1.10%, and the
sweep duly recommends it. Taken alone that is the whole story and conformal
loses.

It is the wrong reading, for a reason this project has already paid for. 2.32
measured two independent 3-run benchmarks of an **identical** configuration and
got medians 8 points apart; 2.18 measured 18.6% variation between repeated
executions of the same plan. A 0.07-point difference is far below anything the
measurement apparatus can resolve. What *is* resolvable is a 5-point swing from
-4.04% to +1.17%, and that swing belongs entirely to `fixed_z`.

The two bounds are therefore not "1.17 versus 1.10". They are:

- **`fixed_z`**: correct only if `min_relative_gain` is set to 0.20 and not
  0.10. At 0.10 it loses 4% and its worst query is 1.73x native. The right
  value cannot be determined by execution, because the differences between
  candidate values are smaller than the noise.
- **`conformal`**: the same answer at every value of that knob, because the
  binding constraint is a quantile measured from the model's own residuals
  rather than a threshold someone picked. Worst case 1.07x instead of 1.73x.

One of these has a parameter that must be tuned and cannot be. The other
measured its own.

That is also the deeper reason `fixed_z` is fragile here: `z * sigma` treats
the ensemble spread as a calibrated standard deviation, and 2.4.5 measured that
spread as understating true error by **3.75x**. `fixed_z` spends `z` on a ruler
with the wrong units. Conformal never asks what the spread means -- it takes
the empirical quantile of the residuals and holds whatever shape they have.
`uncertainty_scale = 3.75` is not a number to correct; it is the reason not to
build a gate on `sigma` in the first place.

### What changed

`app.calibrate` sweeps the bound's *form* as one more axis of the grid, so
`fixed_z` and `conformal` compete on the same held-out rows under the same
dominance rule that already chooses `z` and `min_relative_gain`. Bundles with
no fitted conformal gate contribute no conformal rows and behave exactly as
before.

`--apply` writes `gate_mode` into `models/gate.json`, and `LearnedOptimizer`
reads it when neither a constructor argument nor `GATE_MODE` was given.
Thresholds and the bound they were swept under have to travel together: a
`min_relative_gain` chosen against a conformal bound is not the right threshold
for a fixed-z one. Writing one without the other is 2.28's failure again --
calibrating a configuration nobody runs.

Precedence is explicit argument, then `GATE_MODE`, then the gate file, then
`fixed_z`. The env var deliberately outranks `models/gate.json` here, unlike
`confidence_z`: an operator exporting `GATE_MODE` is overriding the method
itself, and a stale sweep should not quietly undo that.

### The general shape of this bug

Nothing here was missing. The algorithm was implemented and correct, the model
fitted it on every training run, the README documented it, the settings UI
exposed it, and a thoughtful comment explained why it was not on by default.
The gap was that no measurement was ever attached to the decision, so the
default became permanent by inertia and the comment made that look deliberate.

The audits in this project mostly ask "does this code work". This one came from
asking "is this code *reached*", which found `pg_stat_statements` never created
(2.36), two hint families switched off (2.27), and now a safety bound fitted on
every run and consulted on none.

## 2.43 Every calibration this project ever ran measured the wrong policy

2.41 concluded the model's ranking on unseen queries loses, and 2.42 measured
the two gate bounds against each other on that basis. Both are correct about
what they measured. Both measured `greedy`.

The server runs `pairwise_rank`.

### How the disagreement survived being fixed twice

`retrain.served_policy()` exists *because* of this class of bug. Its docstring
says so:

> It used to hardcode `risk_averse` while `docker-compose.yml` sets
> `SELECTION_POLICY: pairwise_rank` and the code default is `greedy` -- three
> policies named in three places. [...] 2.22 names the pattern: a constant that
> duplicates a decision made elsewhere will eventually disagree with it. This
> is the fourth instance.

The fix was to read the environment instead of restating the value. That
removed the *hardcoding* and left the *disagreement*, because of where the
environment variable lived:

```yaml
  backend:
    environment:
      SELECTION_POLICY: pairwise_rank
```

On the `backend` service. So the variable exists inside that container and
nowhere else -- and `app.calibrate`, `app.retrain` and `app.benchmark` are run
from the host, where it is unset and the code default applies. `served_policy()`
returned `greedy` on the host while the container served `pairwise_rank`, and
`app.calibrate --policy` defaulted to the literal string `"greedy"` besides.

Every gate ever written to `models/gate.json` was swept under a policy the
server does not use.

### The two policies differ in sign

If they were near-neighbours this would be a tidiness complaint. Measured on
the same held-out rows, same bundle, same candidate sets:

| | deployed, 8 held out | challenger, 15 held out |
|---|---|---|
| native | 100.0% | 100.0% |
| oracle (best candidate) | 86.0% | 83.6% |
| `greedy` | 103.9% | 104.1% |
| `thompson` | 102.7% | 109.9% |
| `risk_averse` | 102.0% | 103.7% |
| **`pairwise_rank`** | **92.7%** | **95.1%** |
| random pick | 445.8% | 163.7% |

Three policies land *above* 100% -- worse than leaving PostgreSQL alone. The
fourth captures **30%** of the available headroom on queries it has never seen.
It is the only one that beats native, and it was the only one never calibrated.

The gate sweep says the same thing. Wide open, on the challenger's 15 held-out
queries:

| policy | deviates | of those regress | worst | net |
|---|---|---|---|---|
| `greedy` | 87% | 38% | 1.72x | **-4.45%** |
| `thompson` | 73% | 45% | 1.81x | -7.31% |
| `risk_averse` | 87% | 38% | 1.72x | -4.01% |
| `pairwise_rank` | 80% | 50% | 1.70x | **+4.55%** |

Note that `pairwise_rank` regresses *more often* -- 50% against 38% -- and wins
anyway. A rate says how often and not how much (2.19, again).

### So 2.41's verdict was measured on a policy nobody runs

Its central claim -- "on queries it has not seen, this optimizer's plan choices
are currently worse than PostgreSQL's" -- holds for `greedy`, `thompson` and
`risk_averse`, and is false for the policy that serves. The corrected statement
is narrower and much better:

> On unseen queries the served policy beats PostgreSQL by 4.9% and reaches 30%
> of the oracle's headroom. Every ensemble policy loses. The safety gate was
> calibrated on a losing policy and correctly refused to act, which is why the
> measured capture was 0.0%.

2.41 is left standing rather than rewritten, because the reasoning in it is
sound and the failure is instructive: a measurement is only as good as its
agreement with the deployment, and nothing in that section could have revealed
the disagreement. It took asking a different question -- *which policy does the
container actually run* -- which is the same question that found
`pg_stat_statements` never created (2.36) and the conformal gate never used
(2.42).

### Why the ensemble policies lose, and `pairwise_rank` does not

The ranking diagnostic explains it. Within a query's candidate set, the
ensemble's predictions correlate with true latency at Spearman **+0.533**
(median over 15 held-out queries, positive on 14 of them). That is real signal:
a random pick costs 163.7% of native, and the ensemble's argmin costs 104.1%.
It removes 60 of the 64 points between random and native.

But the bar is not "better than random", it is "better than native", and native
sits only 16.4 points above the oracle. The ensemble is being asked to resolve
a band it cannot resolve: good enough to avoid disasters, not good enough to
separate the top few candidates, which is where the entire remaining
opportunity lives.

`pairwise_rank` is trained on exactly that -- orderings between candidates
rather than a latency each -- so it spends its capacity on the comparison that
matters instead of on absolute values that are 2.5x off (2.35) in a way that
mostly cancels when comparing.

That reframes 2.35's "uniformly inaccurate" result. Uniform inaccuracy is
survivable for a *ranker* and fatal for a *regressor used as a ranker*, and
this project has been measuring the latter while serving the former.

### What changed

- `learned.DEFAULT_SELECTION_POLICY = "pairwise_rank"` -- one constant.
- The `SELECTION_POLICY` line is **deleted** from `docker-compose.yml`. The
  lesson from 2.22 is to remove the duplicate, not to synchronise it; exporting
  the variable still overrides both host and container at once.
- Every default that named a policy now reads that constant: `LearnedOptimizer`,
  `app.settings`, `retrain.served_policy`, `benchmark.run`, `benchmark`'s
  `--policy` argument, and `calibrate`'s `sweep`, `evaluate_setting` and
  `measure_settings`. That is **eight** places, which is the measure of how far
  a duplicated decision spreads before anyone notices.
- Three tests assert the deletion stays deleted, that `app.settings` and
  `learned` name the same default, and that `served_policy()` returns it.

The eighth site is the uncomfortable one. `benchmark.run(policy="greedy")` was a
default *argument*, not a missing environment variable, so it applied inside the
container too. **Every benchmark number this project has published was measured
under `greedy`** -- including the held-out 0.0% of 2.37 and the eleven runs
behind it. Those numbers are not wrong about what they measured; they are
answers to a question nobody asked.

### A note on how this kept happening

Each fix in this chain was correct and each was defeated by where the value
lived. Hardcoding `risk_averse` was replaced by reading `SELECTION_POLICY` --
which the host could not see. Compose was given the right value -- on one
service. `served_policy()` was written to centralise the answer -- and three
call sites kept their own literal defaults, because a default argument is not a
lookup.

The pattern is not "someone forgot". It is that a policy is a *deployment*
property being expressed as a *language* default, and Python evaluates those per
call site. The only fix that holds is one name.

The gate has not been re-applied yet, because under the served policy every
setting in the grid is refused by `--max-worst-regression`: the best is
`z=0.0, gain=0.0` at **+4.5% (+188 ms)** with one query landing at **1.70x**
native, against a bound of 1.25x. That is the safety bound working, on a real
opportunity, and the answer is a better worst case rather than a looser bound.

## 2.44 A five-millisecond regression was vetoing a 188 ms win

With 2.43's policy fix in place, the sweep under `pairwise_rank` produced a
real opportunity and then refused all of it:

```
No setting qualified. The closest was z=0.0, min_gain=0.0 (net +4.5%, +188 ms).
  refused by --max-worst-regression: one query ends up 1.70x slower than
  native, bound is 1.25x.
```

The query is `ctl_supplier_rating`, and the regression is **7.8 ms to 13.2 ms**.

### The bound cannot see magnitude

Per-query, under the served policy on the challenger's 15 held-out queries:

| query | native | served | ratio | absolute | verdict |
|---|---|---|---|---|---|
| `ctl_supplier_rating` | 7.8 | 13.2 | **1.70x** | **+5.4 ms** | refuses the gate |
| `gen_07_2w` | 91.8 | 114.8 | 1.25x | +23.0 ms | admitted |
| `shape_top_n_orders` | 514.4 | 627.1 | 1.22x | **+112.7 ms** | admitted |

The rule this bound encodes is the one from 2.19: *a single user-facing query
getting 3x slower outweighs a diffuse average win*. The operative idea is a
person waiting. Expressed purely as a ratio, it refuses a 5 ms loss and admits
a 113 ms one -- exactly backwards, and for a mechanical reason: on a query that
finishes in under ten milliseconds, any regression at all is a large multiple.

`ctl_supplier_rating` is a control query. Its whole purpose is to have no
headroom, and it does not: its *oracle* is also 13.2 ms, so every candidate is
worse than native and the ranker is picking the best of a bad set. The bound was
not protecting anything that could be improved. It was refusing 188 ms of wins
elsewhere over 5.4 ms on a query where nothing better exists.

### The fix, and the reason it is not a loosened threshold

`MIN_REGRESSION_MS = 10.0`: a regression below this does not enter the
worst-case *ratio*. It still enters the regression *rate* -- "how often" and
"how badly" are different questions and only the second is about size (2.19
again) -- and `worst_regression_ms` is reported alongside the ratio, so the two
measures that disagree about which regression is the bad one are both visible.

This is deliberately not a raised `--max-worst-regression`. Raising the ratio to
1.75x would admit this 5.4 ms loss *and* a 900 ms loss on `shape_top_n_orders`.
The floor admits the first and leaves the second bounded at 1.25x, which is the
distinction the original rule was trying to draw.

10 ms is a judgement and is labelled one. What is measured is that it is not
delicately poised. Sweeping the floor from 0 to 200 ms, the recommendation
changes at exactly one point -- 5.4 ms, the size of this regression -- and is
identical everywhere above it:

| floor | recommended | net | worst regression |
|---|---|---|---|
| ≤ 5.4 ms | z=1.00, gain=0.20 | +0.89% | none |
| ≥ 5.5 ms | z=2.00, gain=0.02 | **+3.29%** | 5.4 ms |

Any value from ~6 ms to 200 ms gives the same gate, and that gate's worst
regression *is* the 5.4 ms one -- there is no second, larger regression waiting
behind it. `--min-regression-ms 0` restores the previous behaviour exactly,
which matters because every number published before this was produced without
it.

### What the workload gets

The gate the floor admits is not the wide-open one. It is `z=2.0, gain=0.02`:
**+3.29% net, worst regression 5.4 ms, worst ratio 1.00x on anything material**.
The wide-open setting was worth +4.5% but carried the 1.70x. Trading 1.2 points
of net for a gate whose largest material regression is *zero* is the trade the
bound was written to make -- and it could only be made once the bound could tell
5 ms from 113 ms.

## 2.45 The promotion yardstick was the alphabet

2.43 fixed *which policy* promotion scores. This is about *which queries*.

```python
def _held_out_rows(fraction: float = 0.25) -> dict[str, list[dict]]:
    """The most recent queries' rows, grouped -- a shared yardstick for both models."""
    ...
    query_ids = sorted(by_query)
    return {q: by_query[q] for q in query_ids[-n_test:]}
```

`sorted(by_query)` sorts query **ids**, alphabetically. Nothing in this function
is chronological, and `created_at` is present on every row and unused. "The most
recent queries" was a description of intent that the code never implemented.

### What the alphabet selected

On this workload the last quarter of the alphabet is `gen_21` through `gen_27`
and eight `shape_*` queries -- a group defined by naming convention. Whether
that set is a fair test of a model depends entirely on coincidence.

It was not fair. Measured on the bundles in hand:

| Model | queries in the yardstick it had trained on |
|---|---|
| champion | **13 of 15** |
| challenger | **10 of 15** |

Promotion was comparing two models on a sample both had largely memorised, and
the one that had memorised less of it is the one that looks better on the
out-of-sample fraction. The +20.8% "improvement" that decides promotion is being
read off five uncontaminated queries at most, and the arithmetic does not say
which five.

### Two fixes, and one that was rejected

**Recency is measured.** Queries are ordered by their latest `created_at`, so
the name is now accurate. Rows without a timestamp fall back to the id, keeping
the order total and the selection reproducible.

On the current log this changes nothing -- the newest queries are also the last
alphabetically, because `gen_*` and `shape_*` were added most recently. That is
the coincidence the old code was resting on, and it would have broken the first
time anyone added a query starting with `a`.

**Contamination is preferred against, and reported when it cannot be avoided.**
`_held_out_rows(bundles=[...])` prefers queries that *no* model in the
comparison trained on.

The rejected fix is to require that. The champion held out 8 queries and the
challenger 15, and the two lists intersect in **two** -- `corr_6w_everything`
and `shape_distinct_brands`. Two queries cannot separate two models; a bound
computed on them would be noise wearing a clean label. So the recency yardstick
stands when the clean set is under five queries, and `yardstick_provenance`
records what each model had already seen, which goes into the promotion
decision and out through `app.retrain --status`.

An honest contaminated measurement beats a clean one too small to read. What is
not acceptable is the third option, which is what was there: a contaminated
measurement that looks clean.

### The finding that fell out of fixing it

Scored under the served policy rather than `greedy` (2.43), the champion and
challenger diverge on the metric 2.31 added:

| | mean | worst pick vs best available |
|---|---|---|
| champion | 299.1 ms | **118.98x** |
| challenger | 238.5 ms | 4.88x |

The champion's 119x is `shape_window_rank`: it picks a 91.7 ms plan where the
best candidate runs in 0.8 ms. Worth stating precisely -- native is 280.9 ms, so
that pick still beats PostgreSQL by 3x. It is not a regression, it is a missed
opportunity, and `worst_ratio_vs_best` measures distance from the oracle rather
than danger to the user. Under `greedy` both models score an identical 3.820x on
this metric, so the entire difference between "these two models are the same"
and "one of them leaves 119x on the table" lives in the policy that was not
being scored.

## 2.46 The bound comparison reverses under the policy that serves

2.42 scored `fixed_z` against `conformal` and concluded that conformal was the
better-behaved bound: flat at +1.10% across every value of `min_relative_gain`
where `fixed_z` swung from -4.04% to +1.17%, with a worst case of 1.07x against
1.73x. The argument was about robustness rather than magnitude, and it was a
good argument.

It was also measured under `greedy` (2.43). Re-scored under `pairwise_rank`, on
the same 15 held-out queries, with the absolute regression floor from 2.44:

| bound | min_gain | deviates | of those regress | worst | worst ms | net |
|---|---|---|---|---|---|---|
| `fixed_z` (z=2.0) | 0.00 | 40% | 17% | **1.00x** | **1.0 ms** | **+3.10%** |
| `conformal` | 0.00 | 13% | 50% | 1.96x | **292.9 ms** | **-11.64%** |
| `fixed_z` (z=2.0) | 0.05 | 20% | 33% | 1.00x | 1.0 ms | +0.95% |
| `conformal` | 0.05 | 13% | 50% | 1.96x | 292.9 ms | -11.64% |

Conformal is still flat. It is flat at **-11.64%**.

### The mechanism, as far as 15 queries can show it

`ConformalGate` here is fitted with `normalize=False`, so the bound is
`mean + q` -- a single constant, identical for every candidate. `fixed_z` is
`mean + z·sigma`, which varies per candidate. Per-query, the picks the two
bounds disagree about:

| query | ensemble spread | `fixed_z(2)` | `conformal` | native | served |
|---|---|---|---|---|---|
| `shape_top_n_orders` | **0.266** | 2.013 — blocked | **1.051 — admitted** | 305.5 | **598.4** |
| `corr_5w_premium_supplier` | 0.071 | 1.095 — admitted | 1.247 — blocked | 122.3 | 107.2 |
| `ctl_supplier_rating` | 0.066 | 1.012 — admitted | 1.173 — blocked | 6.0 | 7.0 |

A constant-width interval is too loose where the ensemble is unsure and too
tight where it is confident. `shape_top_n_orders` is the whole -11.64%: a
predicted 0.70x that came back at 1.96x, admitted by a bound that could not see
a spread of 0.266.

**This is weaker evidence than it looks and is worth saying so.** Across all 15
held-out picks, the correlation between ensemble spread and the served ratio is
Spearman **+0.250** -- the right sign and not much more. Splitting at 0.10:
high-spread picks regress 4 times in 7 with a median ratio of 1.288; low-spread
picks regress 2 in 8 with a median of 0.929. Directionally consistent, on a
sample far too small to establish. The honest reading is that the two worst
picks also happened to carry the two largest spreads, and `z·sigma` blocked
both.

### The calibration and the serving distribution differ in the one thing the bound depends on

There is a real tension here worth stating, because both sides are measured.

`optimizer/conformal.py` chose `normalize=False` on evidence: over 2,483
calibration rows, the normalised bound was unusable at every risk level worth
having -- 1.38x at alpha=0.10, numerically meaningless at 0.05 -- because rows
where the eight members agree almost exactly produce a near-zero denominator
that dominates the quantile. The members disagree by 0.026 while the model's
error is 0.106: they agree because they are all making the same mistake.

That is true of the **calibration rows**, which are training rows. On the
held-out queries above, the spreads are 0.018 to 0.334 -- an order of magnitude
larger, which is what `learned._upper_bound` already documents. So the bound is
calibrated on a distribution where its key input is nearly degenerate and
applied on one where it is not.

That is train/serve skew located inside the safety calibration itself, and it
explains both measurements without either being wrong.

### What this changes

**2.42's recommendation does not survive, and no replacement is declared.**
`fixed_z` wins on the served policy by a wide margin, but 2.32 already
established that this workload cannot resolve differences of a few points by
execution, and this is a replay of 15 queries. Naming a winner here would repeat
the mistake 2.42 made -- reading a result as settled when it was conditional on
something unexamined.

What the sweep does is score both and record which it chose, per bundle, under
the served policy. That machinery is the durable part; the current answer is
`fixed_z, z=2.0`, and it is an answer about one model and 15 queries.

The specific follow-up is concrete: refit the conformal gate with
`normalize=True` on residuals whose spreads resemble serving rather than
training, and re-score. The near-zero-denominator failure is a property of the
calibration sample, not of the method, and the sample was chosen before anyone
had measured what serving spreads look like.

## 2.47 The replay could not see the veto it was supposed to model

The benchmark under the served policy reported the headline at 41.0% of
available headroom and held-out capture at **0.0%**, three runs, unchanged. The
replay, on the same model and the same deployed gate, said those held-out
queries should deviate twice and net **+5.38%**.

Both cannot be right, and this project has now spent enough time on
replay-versus-execution gaps (2.30, 2.32) to treat the disagreement as the
finding rather than the noise.

### Reproducing the serving decision, without executing anything

Driving `LearnedOptimizer` over the eight held-out queries with live candidate
generation and printing `last_decision`:

| query | candidates | ranker's pick | ensemble's best | outcome |
|---|---|---|---|---|
| `corr_city_country` | 3 | #1 @ 0.748x | #1 @ 0.748x | **vetoed** |
| `corr_4w_full_correlated` | 10 | #2 @ 1.278x | #4 @ 1.014x | no gain |
| `corr_4w_cancelled_premium` | 11 | #0 @ 0.965x | #0 @ 0.965x | no gain |
| `corr_5w_supplier_chain` | 13 | #7 @ 1.128x | #12 @ 1.020x | no gain |
| `corr_6w_everything` | 18 | #17 @ 1.160x | #0 @ 1.118x | no gain |
| `ctl_country_only` | 3 | #1 @ 0.745x | #1 @ 0.745x | **vetoed** |
| `shape_distinct_brands` | 10 | #9 @ 1.472x | **#5 @ 0.703x** | no gain |
| `shape_anti_join_unsold` | 4 | #1 @ 0.912x | #1 @ 0.912x | **vetoed** |

Eight of eight kept native. Three were killed by `learned._is_unsafe`, which
refuses any chosen plan carrying `Disabled: true` nodes.

### The replay had no way to apply that veto

`_row_to_candidate` rebuilds a candidate dict from a logged row. It
reconstructed `tables_scanned`, `scan_relations`, `join_types` and `total_cost`
from the stored plan -- and not `disabled_nodes`. `plan_extractor.get_plan` sets
that field on every *live* plan; nothing set it on a replayed one.

`featurize` did not care, because it falls back to counting the tree itself:

```python
features["n_disabled_nodes"] = float(
    candidate_plan.get("disabled_nodes") or _count_disabled(raw_plan)
)
```

So the *feature* was always right, every model trained on the correct value, and
nothing looked wrong anywhere. `_is_unsafe` reads the key directly with no
fallback, saw `None` for every replayed candidate, and the safety veto simply
did not exist inside `app.calibrate`.

Counting the same four queries before and after the one-line fix:

| query | replay candidates | with `Disabled` before | after |
|---|---|---|---|
| `ctl_country_only` | 3 | 0 | **1** |
| `corr_city_country` | 3 | 0 | **1** |
| `shape_anti_join_unsold` | 4 | 0 | **2** |
| `shape_distinct_brands` | 10 | 0 | **2** |

With the veto applied, the replay reproduces execution exactly: the deployed
gate on the deployed model deviates on **0%** of its held-out queries and nets
**0.00%** -- which is the 0.0% the benchmark measured eleven times, finally
explained rather than merely observed.

On PostgreSQL 18 this matters more than it would have on 16. 2.29 records that
the version removed `disable_cost`, so a forbidden plan no longer announces
itself with an astronomical cost; the flag is the only signal left.

### The same blindness was in the promotion path, twice over

`retrain._score_bundle` shared `_row_to_candidate` and so shared the veto
blindness, and it had two more divergences of its own:

- it scored **every** logged candidate, including hint families the serving path
  does not offer -- the exact mistake `calibrate._servable` was written to fix,
  and whose cost that function documents (an all-candidates replay had every
  gate losing 10-25% on forced nested-loop plans the live path drops);
- it scored the **baseline row** as a candidate. Serving calls
  `select_plan(candidates, baseline_plan=native)`: the policy chooses among
  candidates and the *gate* decides whether to use the choice. Leaving native in
  the pool let a model score well by "picking" a plan it never had the option
  to pick.

All three are now filtered, so promotion ranks two models on the action space
they will actually be given.

### What was not the problem

Worth recording, because it was the obvious hypothesis and it is wrong.

`_select_against_native` vetoes the chosen plan and returns the baseline; it
does not fall through to the runner-up. That looks like a bug -- one
out-of-distribution candidate discarding every other option for the query --
so it was measured. Walking the ranker's ordering until a candidate clears the
gate, against taking the top pick or nothing:

| | deviates | worst regression | net |
|---|---|---|---|
| top-1 only, z=0.5 gain=0.10 | 4/15 | 10.6 ms | **+1.00%** |
| walk the ordering | 5/15 | 59.8 ms | **-3.00%** |
| top-1 only, z=2.0 gain=0.02 | 4/15 | 0.0 ms | **+6.65%** |
| walk the ordering | 4/15 | 0.0 ms | +6.65% |

Neutral at the good setting and clearly worse at the loose one. The candidates
ranked below the top pick are worse, and letting the gate authorise them admits
marginal plans without finding better ones. Top-1-or-nothing stays.

## 2.48 A tuner that picks the noisiest option, and the fix

Four tools swept the configuration and none of them swept it together:

    app.calibrate    confidence_z, min_relative_gain, gate_mode
    app.model_zoo    the model class
    app.experiment   an A/B of two arms you name
    (nothing)        the selection policy, the action space

Each held everything it was not sweeping at whatever the environment said, and
2.43 is the bill for assuming those are independent. `app.autotune` sweeps
policy x action space x gate on the same held-out rows under the same safety
bounds, and prints one configuration plus the variables to export.

### The first version was wrong, in a way worth keeping

Its first run recommended:

```
  selection policy   : thompson
  confidence_z       : 1.5
  net +11.9% (+177 ms)
```

against `pairwise_rank` at +10.5%. So the deterministic policy the project had
just spent 2.43 establishing as the right one lost, by 1.4 points, to Thompson
sampling.

`thompson` samples one ensemble member per decision. Running the identical
sweep three times:

| run | recommended z | net |
|---|---|---|
| 1 | 0.75 | **+14.16%** |
| 2 | 1.50 | **+10.61%** |
| 3 | 2.00 | **+12.33%** |
| `pairwise_rank` | 2.00 | +10.51%, every time |

Different threshold each time, and a 3.5-point range. The +11.9% was one draw,
and the tuner had compared it against a deterministic policy's exact value.

**Taking a max over configurations is a machine for finding whichever one got
lucky**, and it favours whichever option is noisiest -- the more a
configuration's score varies, the higher its maximum. This is 2.26 ("a single
benchmark run does not measure this system") arriving from the other direction:
there, one run was mistaken for a measurement; here, one run was mistaken for a
measurement *and then maximised over*.

### The fix

- `evaluate_setting` takes a `seed` and passes `random.Random(seed)` into
  `select_index`, so a replay of a stochastic policy is reproducible at all.
- `app.autotune` draws stochastic policies `--repeats` times (default 5) and
  scores them on the **median**, reporting the range beside it.
- A stochastic policy that clears the safety bounds on some draws and not others
  has **not** cleared them. Requiring every draw stops a bound being satisfied
  by luck, which is the one thing a safety bound must not permit.
- The setting reported is the draw nearest the median, not an average of
  settings -- an averaged `z` and `min_relative_gain` may never have been
  measured together.

Re-run:

```
        policy           extra families     z  gain  dev%  reg%   worst    net%  spread
 pairwise_rank                     none  2.00  0.00   33%   20%   1.21x   10.5%
 pairwise_rank              join_method  2.00  0.00   33%   20%   1.21x   10.5%
      thompson                     none  2.00  0.02   33%   20%   1.21x    9.6%  +7.6 to +10.0 over 5 draws
        greedy                     none  2.00  0.02   33%   20%   1.21x    9.2%
   risk_averse                     none  2.00  0.02   33%   20%   1.21x    7.3%
```

`thompson`'s median is 9.6% and its range does not reach `pairwise_rank`'s
10.5%. The ordering now matches 2.43's independent measurement, which is the
first evidence that the tuner is measuring the thing it claims to.

### What it found

**Turning the `resource` family off is worth about 3 points.** `pairwise_rank`
with the default action space scores 7.7%; without `resource` it scores 10.5%.
`INCLUDE_TUNING` has defaulted to `1` for the life of the project. 2.27
measured the switched-off families earning nothing across 24 queries and left
`resource` on; nobody re-measured it after the workload tripled or after the
PostgreSQL 18 upgrade, and a family whose inclusion nobody re-checks is a
decision that has stopped being evidence.

The winner and its runner-up differ only in an action space that changes
nothing, so the tool says so and prefers the narrower one -- extra candidates
are planning round trips at serving time, and a tie should resolve towards
fewer.

### What it deliberately does not do

`--apply` writes `models/gate.json` and **prints** the policy and action space
rather than exporting them. Thresholds are a file this tool owns; what a running
server selects with is not something a tuning run should change behind the
operator's back.

And it does not execute. It is a replay, inheriting every caveat `app.calibrate`
carries -- with the corrections of 2.43 and 2.47 now included. The last line of
its output says to confirm the winner with `app.benchmark --protocol compare`,
because 2.32 measured two identical 3-run benchmarks landing 8 points apart and
a replay difference of a few points is not a result.

## 2.49 A new gate, reported broken on the previous gate's history

Straight after the recalibration in 2.48, `app.diagnostics` said:

```
[  high] gate_promise_not_met
   the calibration promised 20% of deviations would regress; 31% of the last
   1310 matched runs did.
```

The gate had existed for about a minute. All 1310 of those runs were served
under the previous gate, the previous model and -- given 2.43 -- effectively the
previous policy.

`drift_report` compares `expected_regression_rate` against the observed rate
over `stats.served_vs_native`, which returns the whole matched history. It is a
genuinely good check: a promise nobody verifies is decoration, and this is the
only thing in the project that verifies one. But it had no notion of *which*
gate produced the history it was reading, so every recalibration inherited the
verdict on its predecessor -- and the worse the old gate was, the louder the new
one was declared broken.

The fix is small and in two halves. `apply_gate` records `applied_at`;
`served_vs_native` takes a `since` and restricts `decision_quality` to runs from
that moment. The run series and the totals stay unfiltered, because the
dashboard's history is a history.

The second half is the part worth insisting on. A gate file written before
`applied_at` existed has no cutoff, and the warning now says so rather than
quietly reverting to the whole log:

> ...though this gate file records no `applied_at`, so some of those runs may
> predate it and belong to a different gate entirely

That distinction -- "measured, and here is the number" against "could not
measure, here is the closest thing" -- is the same one `apply_gate` had to learn
when it was writing `0.0` for regression rates nobody had computed. It keeps
coming up because the failure is comfortable: both readings produce a number,
and only one of them is an answer.

### The cost of the honest version

A freshly applied gate now has **no** history and cannot be judged at all until
`MIN_RUNS_FOR_DRIFT` runs accumulate under it. That is worse than a
misattributed verdict only if the misattributed verdict was worth having, and
it was not: the previous behaviour reported a defect in the gate that had just
been installed to fix that defect.

## 2.50 The same divergence, in the two places left

2.47 fixed the replay's blindness to the safety veto in `app.calibrate` and
`retrain._score_bundle`. Auditing for the pattern rather than the instance found
it twice more.

### `train._evaluate_selection`

This produces `headroom_captured_vs_oracle` -- the number `app.retrain` prints
and the dashboard shows. It built its candidate list as:

```python
candidates = [_row_to_candidate(r) for r in rows]
```

Every logged row, the **baseline included**, and no `_servable` filter. Two
consequences, both of which flatter:

- Native is not a candidate. Serving calls
  `select_plan(candidates, baseline_plan=native)`: the policy chooses among
  candidates and the *gate* decides whether to use the choice. With native in
  the pool a policy scores well by "picking" a plan it was never offered -- and
  the oracle, the ceiling every headroom figure is divided by, silently becomes
  `min(candidates + [native])`.
- The log records everything ever collected, and collection explores wider than
  serving does. That is `calibrate._servable`'s entire subject.

So the headline the retrain prints described an action space no query is given.
Filtered now, all three ways.

### `main._optimizer_from`

```python
    return LearnedOptimizer(
        ...
        # Not part of `gate` above: which bound to apply is a plain choice with
        # no calibrated-file default to defer to, unlike the thresholds.
        gate_mode=values["gate_mode"],
```

The comment was true when written and stopped being true in 2.42, when
`app.calibrate` learned to sweep the bound's form and write the winner into
`models/gate.json`. Passing it unconditionally means the constructor argument
always wins, so the API server never read the swept value.

A calibration the server ignores: 2.43's policy, 2.42's unused conformal gate,
and now this. `gate_mode` joins the thresholds and is forwarded only when
someone actually chose it in the settings panel -- which is also correct on its
own terms, since a `min_relative_gain` chosen against a conformal bound is not
the right threshold for a fixed-z one.

### And a check, so the next one is loud

`app.autotune` records `calibrated_with_families` in the gate file, and
`app.diagnostics` now compares it against what the deployment offers. On the
current configuration it fires immediately:

```
[medium] gate_calibrated_for_another_action_space
   the gate was swept over cardinality, join_order, operator_toggle but the
   server also offers resource.
```

Which is correct: `app.autotune` recommended dropping the `resource` family and
nothing has exported `INCLUDE_TUNING=0` yet. The gate is measuring a system the
server is not running, and now it says so instead of being discovered a month
later.

Three checks of this shape exist now -- policy (2.43), action space, and the
drift window (2.49) -- and they share a rule: a configuration file that records
what it was measured under can be checked against reality, and one that does not
cannot. Every one of them was added after the corresponding mismatch had already
been serving for months.

### What the feature audit found, and why nothing was changed

Counting how often each of the 64 features is split on across the 8 ensemble
members, over 411 servable candidate vectors:

| | count |
|---|---|
| constant across every row | 3 (`has_hint`, two `*_index_scan`) |
| never split on by any member | **16** |
| split on 5 times or fewer | 1 |

A quarter of the feature vector is dead weight that still costs a `featurize`
call on the serving path. `has_hint` is the clearest case: it is always 1.0,
because candidates always carry a hint and native is not a candidate.

Nothing was removed, and the reason is worth stating rather than leaving as an
omission. §2.39 measured `featurize` at 3.08 ms of a 27.5 ms request, so
deleting a quarter of the features saves under 1 ms -- around 3% -- while
`has_hint` is still load-bearing on the legacy absolute-latency path, which
*does* score native alongside the candidates. Trading a real behavioural risk on
one code path for 0.8 ms on another is a bad trade, and "measured, not worth
acting on" is a result. It is recorded here so the next person does not have to
measure it again.

## 2.51 Zero becomes forty-six

The benchmark, re-run under the configuration `app.autotune` chose, on the model
promoted after the yardstick fix, with the veto-aware gate:

```
=== protocol: compare (3 runs) ===
headroom captured        41.3%   range 38.9% to 43.0%   runs 38.9, 41.3, 43.0
held-out captured        45.7%   range 19.3% to 50.4%   runs 19.3, 45.7, 50.4
curated only             33.0%   range 20.8% to 34.2%   runs 34.2, 20.8, 33.0
```

The middle line is the one this project exists to produce. It had read **0.0%**
on every run ever measured -- eleven of them across three independent sets
(2.37), and three more this session (2.41's benchmark). It now reads **45.7%**.

### What actually changed

Nothing in the model architecture, the features, or the action space. Six
corrections, each of which had been silently wrong for months:

| | |
|---|---|
| 2.43 | the gate was calibrated under `greedy`; the server selects with `pairwise_rank`, and on held-out queries those land at 104.1% and 95.1% of native |
| 2.44 | a 5.4 ms regression on an 8 ms query read as 1.70x and vetoed a gate worth +188 ms |
| 2.45 | the promotion yardstick was the alphabet, and the challenger had trained on 10 of its 15 queries |
| 2.42 | the conformal bound was fitted every run and consulted on none |
| 2.47 | the replay could not see the safety veto, so it scored plans the server refuses |
| 2.50 | the training eval scored native as a candidate, and the API ignored the swept `gate_mode` |

The system was capable of this the whole time. What it lacked was any
measurement that agreed with what it was running.

### Read the third line too

`curated only` -- the 18 correlation traps, 6 controls and 8 shape queries,
written by hand against PostgreSQL's independence assumption and never filtered
on whether this optimizer could win them -- captures **33.0%** against the
headline's 41.3%.

That gap is the selection effect, and this is the first time it has been
measured rather than warned about. 27 of the 59 workload queries were generated
and then kept *because* the best plan in the served action space beat native by
at least 8%, so the headline is measured partly on a benchmark chosen for having
something to capture. Eight points, and the curated number is still a third of
the available headroom.

### And the caveat the tool itself insists on

The held-out range is **19.3% to 50.4%** across three runs. That is enormous,
and it is the same instrument 2.32 caught giving two identical configurations
medians eight points apart. The summary line the benchmark prints says a
difference smaller than 4.2 points should not be argued about; a 31-point spread
on the headline metric says the honest claim is "clearly not zero", not "45.7%".

What is safe to say: on queries this model never trained on, the served
configuration is now taking a substantial fraction of the available headroom,
and every prior measurement of that quantity was 0.0% because the thing being
measured and the thing being run were different systems.

## 2.52 The tuner re-created the bug it was built to prevent

`app.autotune` exists because 2.43 found the gate calibrated under one policy
and the server selecting with another. Run against the promoted model, it said:

```
=== best configuration ===
  selection policy   : greedy
  net +13.1% (+228 ms)
  the runner-up (greedy with resource, +13.1%) is close enough that this replay
  cannot separate them.
```

`pairwise_rank`, the running policy, scored **12.3%**. A gap of 0.8 points, and
`--apply` duly wrote:

```json
  "calibrated_with_policy": "greedy",
```

into a gate the server would apply while selecting with `pairwise_rank`. The
instrument built to prevent 2.43 committed 2.43, and printed a line admitting
it could not resolve the difference it was acting on.

The mismatch check added an hour earlier caught it in seconds:

```
[  high] gate_calibrated_for_another_policy
   the gate was swept under `greedy` and the server selects with `pairwise_rank`.
```

### Why the existing tie rule did not stop it

`calibrate.TIE_FRACTION` is 5% relative, about 0.65 points on these numbers, so
0.8 points cleared it. That threshold is right *within* a sweep, where every row
is the same policy and a tie means "these thresholds are equivalent". Across
policies it is wrong twice over:

- **It is far below what execution can resolve.** The benchmark prints its own
  floor from the spread of its runs, and this session's read *"a difference
  smaller than 4.2 points is inside this sample's own spread"*. 2.32 measured
  two identical 3-run benchmarks landing **8 points** apart.
- **Switching is not free.** It costs a recalibration, a re-benchmark, and a
  window in which the gate file and the server disagree.

So `MIN_POLICY_SWITCH_POINTS = 2.0`: a challenger must beat the *running* policy
by two percentage points before the tool recommends changing it. Deliberately
looser than the 4.2 execution could resolve -- this stops the flips that are
obviously noise without pretending replay settles what execution cannot. The
report says when it declined:

> kept the running policy over `greedy`, which scored higher by less than this
> replay can resolve. Switching costs a recalibration and a window where the
> gate and the server disagree; the incumbent has to be beaten, not matched.

A tool that silently declines to change something is worse than one that
changes it, because the reader cannot tell it considered the alternative.

### The general shape

An optimiser that maximises over configurations will find whichever one got
lucky. 2.48 caught that in its stochastic form -- `thompson`'s best draw beating
`pairwise_rank`'s exact value -- and fixed it with repeats and a median. This is
the same error in its deterministic form: two policies whose true difference is
smaller than the measurement, where whichever lands higher wins by construction.
The repeat-and-median fix does not touch it, because neither policy is noisy;
the *comparison* is.

The general rule this project keeps arriving at from different directions: a
selection procedure needs to know the resolution of its own instrument, and act
only on differences larger than it.

### A cost worth stating

Turning the `resource` family off (2.48) had an effect nobody planned. The
sweep reads `latest_run_only=True` -- the most recent run per query, so a
comparison is not contaminated by measurements taken months and workloads apart.
The benchmark that confirmed the new configuration ran with
`INCLUDE_TUNING=0`, so its rows contain **no `resource` candidates at all**, and
the next sweep found the action-space column entirely inert: all four spaces
scored identically for a given policy.

**Switching a family off removes the evidence needed to switch it back on.**
The measurement that justified it (10.5% against 7.73%, and no qualifying gate
at all for `join_order + cardinality + resource`) is now historical, and
re-testing it requires deliberately collecting with the family on again. That
is a property of any system that learns from what it chose to do -- the standard
bandit problem -- and it is worth naming here because nothing in the pipeline
currently explores against its own configuration.

## 2.53 A setting with six lines of help text and no effect

Auditing the settings surface the same way -- *is this reached?* rather than *is
this correct?* -- against every field the dashboard declares:

```
group_by_template   0 references outside app/settings.py
```

It is referenced, as it happens: `logging_store.py` reads
`GROUP_BY_TEMPLATE` from the environment. **At import.** The settings panel
writes `models/settings.json`, and `logging_store` never consults it, so
toggling *"Treat parameters as the same query"* in the dashboard did nothing at
all.

`main._apply_settings` exists precisely to push a settings change into the
running system. It syncs the optimizer, the regression guard, the decision cache
and fifteen `planner` flags. It did not sync this one, and there was no way to
notice: the panel showed the new value, `GET /settings` returned the new value,
and the behaviour was the old one.

The setting is not decorative. Its own help text explains that every per-query
mechanism -- the regression guard, the caution applied to unseen queries, the
decision cache, the trend chart -- is keyed on query identity, and that keying on
literal text makes a parameterised application look like thousands of one-off
queries. Someone turning it off to debug exactly that would have seen no change
and concluded something else was wrong.

### The check, rather than the fix

The fix is one line. The interesting part is that this is the fourth instance of
the same shape in one session -- the conformal gate fitted every run and
consulted on none (2.42), two hint families switched off and never re-measured
(2.27), a swept `gate_mode` the API ignored (2.50) -- so the fix worth having is
the one that catches the fifth.

`test_every_declared_setting_is_actually_applied` reads the source of
`_apply_settings` and `_optimizer_from` and asserts that every name in
`settings.FIELDS` appears in one of them, or in an explicit exemption list that
has to carry a reason:

```python
NOT_APPLIED_AT_SERVE_TIME = {
    # Read by `app.train` when the conformal gate is fitted; changing it takes
    # effect on the next training run, not on the next query.
    "conformal_alpha",
    ...
}
```

Grepping source text from a test is crude and it is the right crudeness here:
the failure mode is a name that exists in one file and nowhere else, which is
exactly what a text search detects and a behavioural test does not. A second
test asserts every exemption names a field that still exists, so an exemption
cannot outlive the setting it was carved for and silently re-open the hole.

## 2.54 The panel said 1.0 while the gate applied 1.5

Smoke-testing the running API rather than the test suite -- start `uvicorn`,
call the endpoints a browser calls:

```
GET /settings            confidence_z: 1.0    min_relative_gain: 0.05
GET /model/diagnostics   confidence_z: 1.5    min_relative_gain: 0.02
```

Both are correct and they disagree. `app.settings` declares 1.0 and 0.05 as
defaults; `LearnedOptimizer` overrides both with the measured gate in
`models/gate.json` unless someone explicitly chose them, which is exactly what
`_optimizer_from` is written to do and 2.42 extended to `gate_mode`. The
optimizer applies 1.5. The settings panel reports 1.0.

`app/main.py`'s own header describes this failure as fixed:

> `app.settings` persists every dashboard change to models/settings.json
> precisely so it survives a restart, but nothing outside `PUT /settings` ever
> read it back. The panel therefore reported one configuration while the process
> ran another, and not harmlessly [...]

That was the version where the panel wrongly *won*. This is the same sentence
with the arrow reversed: the calibration correctly wins, and nothing says so. A
user reading the dashboard to find out what the safety gate is doing gets a
number that has been overridden, with no indication that it has been.

### The fix is to report three things, not two

`GET /settings` returns `values` (what is stored), `in_force` (what the live
optimizer holds), and `overridden` (the names where they differ), plus
`gate_source`. The dashboard renders a line under any control whose value has
been superseded:

> **In force: 0.02** — the measured gate in `models/gate.json` is being applied
> instead of this value. Changing this setting overrides the calibration.

Warning-coloured rather than error-coloured, because the calibration winning is
correct behaviour and only the silence about it was the bug.

### Two things this found on the way

The frontend has a test asserting every CSS class used by a component has a
rule, and it caught `setting-overridden` immediately. That is the same species
of check as the settings-reachability test in 2.53 and the gate-provenance
checks in 2.43 and 2.50 -- cheap, mechanical, and aimed at *is this connected*
rather than *is this correct*. This project keeps finding that the second
question is well covered and the first is not.

And the endpoint smoke test is what surfaced it at all. 1219 backend tests and
142 frontend tests pass without noticing, because every one of them either
constructs the optimizer directly or mocks the endpoint. Nothing asked the
running server what it would tell a browser.

## 2.55 JOB: from nothing to within 1% of the oracle

> **Superseded by 2.80.** The 99.2% here was measured against a ceiling 27 ms
> below the served plan. Re-collected under the wide action space the same model
> serves the same plans and captures **91.0%**, because the ceiling dropped
> 176 ms. Read that section with this one.

`docs/JOB_RESULTS.md` has stood for most of this project's life as its most
useful result and its worst one: on the real Join Order Benchmark, **75% of the
latency provably available and none of it captured**. Every selector returned
the native plan. It was the only test of whether any of this generalises past a
six-table toy schema, and the `job` database did not survive the PostgreSQL 18
upgrade, so nothing built since had been tested against it.

Reloaded from `data/job/csv`: 8.7 GB, 73.0M rows, 22 tables. Re-collected with
**1614 executions across 30 queries** against the original's 194 across 8, and
**8 held out** against its 2.

| selector | avg latency | headroom captured |
|---|---|---|
| native PostgreSQL | 3960.2 ms | — |
| cost heuristic | 2285.3 ms | +42% |
| oracle (best candidate) | 695.6 ms | 100% |
| **`risk_averse`** | **713.7 ms** | **+99.4%** |
| `pairwise_rank` | 722.8 ms | +99.2% |
| `greedy` | 726.9 ms | +99.0% |
| `thompson` | 863.9 ms | +94.8% |

Held-out native averages **3960 ms** against the original's **3703 ms**. Same
database, same benchmark, same expensive-query regime -- which is what makes
this a comparison rather than a different measurement.

**All four policies clear 94%, and the top three sit within 0.4 points of each
other.** That matters more than which one leads. One policy at 99% invites the
question of whether it got lucky; four independent selection rules landing
within half a point of the oracle is a property of the candidate set and the
model, not of the rule reading them.

### Where it comes from, because an aggregate that high is a question

Two expensive queries dominating a latency-weighted sum would produce the same
number as a near-perfect model, and 2.40 is this project's own precedent for the
former. So, per held-out query:

| query | native | oracle | served | vs native | captured |
|---|---|---|---|---|---|
| `17c` | 21603.7 | 1963.3 | **1963.3** | **0.09x** | **100.0%** |
| `17d` | 8822.2 | 2447.9 | **2447.9** | **0.28x** | **100.0%** |
| `4b` | 64.6 | 33.2 | **33.2** | **0.51x** | **100.0%** |
| `1b` | 61.4 | 0.1 | 0.2 | 0.00x | 99.8% |
| `6c` | 4.7 | 4.4 | 4.9 | 1.04x | regression, +0.2 ms |
| `2b` | 463.6 | 460.9 | 522.7 | 1.13x | regression, +59 ms |
| `2d` | 586.1 | 571.7 | 715.4 | 1.22x | regression, +129 ms |
| `5b` | 75.3 | 83.3 | 95.0 | 1.26x | nothing to win |
| **total** | **31681.6** | **5564.8** | **5782.7** | **0.18x** | **99.2%** |

On three of the eight it picks the **oracle plan exactly** -- the same plan,
not a close one -- and lands within 0.1 ms of it on a fourth. `17c` goes from
**21.6 seconds to 2.0**, an 11x speedup, and is 75.8% of all the saving. Top-1
exact-oracle accuracy is **3 of 8**; the aggregate is high because the two it
gets exactly right are the two that cost 21.6 and 8.8 seconds.

**And four of the eight regress**, totalling **-208.2 ms** against 26,099 ms
saved. Three of those are queries with almost nothing to win: `2b` had 2.7 ms of
headroom, `2d` had 14.4 ms, and `5b`'s oracle is *slower than native*, so no
candidate beats PostgreSQL and any deviation loses by construction.

> **That last sentence is wrong, and 2.56 is why.** It is true of the action
> space these numbers were collected under and false of the query. Re-collected
> with scan methods, `5b` has **9.4%** available (oracle 68.2 ms against 75.3 ms
> native), `2b` goes 0.6% -> **34.0%**, and `2d` 2.5% -> **40.0%**. The excuse
> offered here for three of the four regressions -- that there was nothing to
> win -- does not survive widening the search. What survives is that the model
> picked badly on queries where the margin was small, which is a different and
> less comfortable claim.

So the honest description is: **half the held-out queries get slightly worse,
two very expensive ones carry the aggregate, and the trade is 26 seconds saved
against 0.2 seconds lost.** That is the right trade for a benchmark about
expensive joins, and it is not a claim that the model is right about every
query.

**These are ungated numbers.** `_evaluate_selection` reports what each policy
picks with no confidence gate applied. The gate exists precisely for `2b` and
`2d` -- a predicted win smaller than the model's own error -- and would decline
most of them. 99.2% is the ceiling the gate chooses from, not what a deployment
serves.

### Three things worth extracting

**`risk_averse` leading is 2.5 reproduced.** That section found that on
data-starved JOB every policy trusting its own point predictions collapses and
only `risk_averse` survives, because it explicitly penalises candidates the
ensemble disagrees about. Four years of project time later, on a rebuilt
database and a corrected pipeline, the same policy leads for the same stated
reason. `thompson` trailing is that finding from the other side.

**The cost heuristic beats native here** -- 2285 ms against 3960 -- which is the
opposite of the 27-query intermediate run, where it was worse. Adding `17c`, a
21-second query the heuristic gets right, moved it. That is a fair warning about
eight-query samples in both directions.

**This is the strongest evidence that the corrections were real.** 2.51 showed
held-out capture going from 0.0% to 45.7% on the synthetic workload, and the
obvious objection is that twelve corrections to a *measurement apparatus* might
produce a better-looking number rather than a better system. JOB is a different
database, a different schema, a different benchmark, and a workload nobody here
wrote. It moved from 0% to within 1% of the oracle in the same week.

### What is still not tested

**The smallest thirty is where this stops.** JOB goes to 17-table joins and the
candidate sampler has never been pointed at them. `17c` was the expensive end of
what was tested and took twenty minutes to collect, because most of its
candidates hit the 120 s statement timeout. Eight held-out queries is more than
two and still few.

## 2.56 The ceiling was the problem, not the aim

(2.80 re-measured this under the wide space: 91.0%, with the served latency
unchanged. The paragraph below is what predicted it.)

2.55 reports the model capturing **99.2%** of the available headroom on JOB, and
that framing invites the wrong conclusion: that selection is finished and the
remaining work is elsewhere. The oracle is the best plan **in the action space
that was collected**, so 99.2% of it says nothing about how good that space was.

Probing the four expensive held-out queries with a wider search -- more join
orders, and every opt-in family switched on -- found something better than the
recorded oracle on **all four**:

| query | native | oracle on record | wider search | winning hint |
|---|---|---|---|---|
| `17c` | 21603.7 | 1963.3 | **1043.4** | `Leading(n ci mk mc t cn k)` |
| `17d` | 8822.2 | 2447.9 | **2232.9** | `Set(enable_nestloop off)` |
| `2d` | 586.1 | 571.7 | **463.6** | `SeqScan(mc)` |
| `2b` | 463.6 | 460.9 | **311.0** | `SeqScan(cn)` |

`17c` drops another **47%** -- 20.7x faster than native rather than 11x. And
`2b`, which 2.55 describes as having "2.7 ms of headroom", has **153 ms**: a
third of its runtime, reachable by one `SeqScan` hint. That sentence was true of
the action space and false of the query, which is a distinction this project has
been careless about.

### Cause one: the join-order sample was eight

`generate_candidates(max_order_candidates=8)`. A 7-table query has 5040 join
orders, so the search covered **0.16%** of the space:

| tables | orders | 8 candidates covers |
|---|---|---|
| 2 | 2 | exhaustive |
| 4 | 24 | 33% |
| 7 | 5040 | **0.16%** |

A fixed sample covers exponentially less as the query widens, and it runs out
exactly where join order starts to matter. The winning hint for `17c` is a pure
`Leading(...)`, so this was the binding constraint and not a missing family.

**And the budget made it worse by trusting the cost model.** `budget_for` scaled
the candidate count by PostgreSQL's estimate alone. `17c` is priced at **4409**
-- the same tier as a query costing 3900 and finishing in 400 ms -- and takes
**21.6 seconds**. So the search was starved on precisely the queries whose
estimates are most wrong, which are precisely the queries with the most
headroom. A system built because cost estimates are unreliable was using them to
decide how hard to look.

`budget_for` takes join width now. `17c` goes from 20 candidates to 80, and
80 candidates measured **352 ms of planning -- 1.63%** of that query's runtime.
The multiplier is bounded at both ends: it does not apply below 200 cost units,
where a wide join has nothing to win anyway, and the total is capped at 96,
because 108 hints on `17c` already collapse to 25 distinct plans and the plan
space becomes the constraint before the sample does.

### Cause two: 39% of the action space had no name

`SeqScan(...)` wins on `2b` and `2d`, and `hint_family` classified it as
`other`. So did `IndexScan`, `BitmapScan`, `IndexOnlyScan`, their negative
forms, `Memoize`, `NoMemoize`, `Parallel` and `DisableIndex` -- **37 of 96
hints** with the full action space enabled.

`other` is in no `_servable` allowed set. Every one of those candidates would
have been generated and served while being silently dropped from every replay:
`app.calibrate`, `app.autotune`, `retrain._score_bundle`. A gate calibrated over
a strictly smaller action space than the one being served, which is 2.47's bug
in a new place.

It was latent because those families all default to off, so it would have bitten
the first person to turn one on. And `tests/test_family_guard.py` did not catch
it: it asserts the Python and SQL classifiers agree, and they did -- **both said
`other`**. Agreement on a wrong answer is still agreement.

Three families named (`scan_method`, `memoize`, `parallel`), `_servable` and
`diagnostics._served_families` updated to admit them under their flags, and a
test that no hint `generate_candidates` can produce classifies as `other`.
`app.autotune` sweeps `scan_method` as a fourth optional family, so whether it
earns its planning cost is measured per database rather than assumed off.

### Cause three: collection was narrower than serving

Fixing the first two exposed a third, and the comment three lines above it in
`app.collect_data` is about exactly this class of bug:

> The join graph must match what serving uses, or training data is gathered over
> a different action space than the one the model is later asked to score -- the
> train/serve skew README.md warns about.

Only the join-graph half had been done. The call took `generate_candidates`'
bare defaults for everything else -- `max_order_candidates=8`, every opt-in
family off -- while `planner.candidate_hints` forwards the module flags and a
width-scaled budget. After the budget fix, serving plans **90 candidates** for
`17c` where collection gathered 20, and the model would have been asked to score
join orders it had never seen.

Collection may legitimately be *wider* than serving: it gathers the forced
join-method family that `budget_for` often truncates, and `calibrate._servable`
narrows the replay back down. Narrower is the direction that hurts, and it is
the failure 2.9 measured -- 17 of 25 queries into the safety veto. Collection
forwards the same flags and the same `budget_for` now, and a test asserts the
invariant by comparing the two functions' source.

That fix also surfaced a latent `UnboundLocalError`: `planner` was imported
inside an `if` in `collect()`, which makes it a local name for the whole
function, so every new `planner.X` reference would have failed on any call
passing `include_join_methods` explicitly -- which is what `collect_data_job`
does.

### And the width fix had the same bug in it

Widening the search costs planning time on every query it touches, so the first
version gated the multiplier on estimated cost -- `WIDTH_MULTIPLIER_MIN_COST`.
Measured on the synthetic workload, that gate does not work, and the two numbers
that break it sit on opposite sides of every possible threshold:

| query | estimated cost | measured runtime |
|---|---|---|
| JOB `17c` | 4409 | **21603 ms** |
| synthetic `corr_6w_everything` | 12954 | **144 ms** |

The cheaper-costed query takes **150x longer**. Widening `17c` cost 352 ms of
planning against 21.6 seconds -- 1.63% -- and found a plan 47% faster. Widening
`corr_6w_everything` moved its planning from 48 ms to 63 ms against a 144 ms
query, taking overhead from **33% to 44%**. Same multiplier, opposite verdicts,
and cost is *anti-correlated* with runtime across the pair, so no threshold on
it separates them.

Which is the bug this section is about, one level up: I fixed a budget that
trusted the cost model by adding a gate that trusted the cost model.

`budget_for` takes `observed_native_ms` now, read from the latency table the
regression guard already keeps, and declines to widen anything measured under
500 ms whatever its shape or estimate. No history still widens -- first sight of
a query is when exploration is worth most, and one over-wide search is cheap
against never finding the plan.

| scenario | budget |
|---|---|
| JOB `17c`, cost 4409, 7 tables, measured 21603 ms | **80** |
| synthetic 6-way, cost 12954, 6 tables, measured 144 ms | **20** |
| synthetic 6-way, no history | 60 |
| 3-table, cost 500, measured 27 ms | 10 |

The synthetic workload is back to its old budget on every query it has measured,
and JOB gets the wider search. That is the separation cost could not make.

### Confirmed by execution

Re-collecting `2b` and `2d` under the widened space, and reading the oracle back
out of the log:

| query | native | oracle before | oracle after | headroom |
|---|---|---|---|---|
| `2b` | 463.6 | 460.9 | **293.6** | **0.6% -> 36.7%** |
| `2d` | 586.1 | 571.7 | **386.3** | **2.5% -> 34.1%** |

Both won by a single `SeqScan(...)`. These are the two queries 2.55 describes as
having "almost nothing to win", and which *regressed* in that evaluation. A
third of their runtime was available the whole time, in a family that was never
collected.

### What this does to the earlier numbers

It does not invalidate them; it reframes what they measure. 99.2% of headroom
captured is still 99.2% of headroom captured -- against a ceiling now known to
be roughly 900 ms too high on `17c` alone. The model was choosing well among the
options it had, and the options were the constraint.

That is a more encouraging position than it sounds, because action-space work is
cheap and bounded, while model work is neither. It is also the third time this
session that the binding constraint turned out to be something other than the
model: the policy (2.43), the measurement (2.47), and now the candidates.

## 2.57 Three CREATE STATISTICS statements remove 98% of the opportunity

`dbadvisor/extended_stats.py` finds correlated columns and reports them. The
roadmap's argument for acting on them rather than reporting them was that
`CREATE STATISTICS ... (dependencies)` fixes the estimate at its source, which
helps every query against those columns forever -- including ones this optimizer
never sees -- and *shrinks the headroom the optimizer is chasing*.

That last clause was a hypothesis. It is now measured, and it is larger than
expected.

### The advisor could not see one of the two designed traps

`recommend()` returned two objects and `users(city, country)` was not among
them -- one of the two correlations `data/schema.sql` exists to contain,
measured at **63.1%** available headroom in 2.9.

Evaluated directly it passes easily: q-error **1.89 falling to 1.00**, an
improvement of 0.47 against a 0.25 threshold. It was never evaluated.
`correlated_pairs` iterated `sorted(by_table.items())` and returned the moment
it hit `max_pairs=24`, so of 58 candidate pairs it took the first 24
alphabetically:

```
order_items 6, orders 10, product_suppliers 6, products 2, then nothing
```

`suppliers` and `users` were never looked at. The advisor was systematically
blind to the end of the alphabet.

Round-robin across tables now, largest first: every table contributes its best
candidate before any table contributes a second, and the budget goes to big
tables first because a dependency on 500k rows costs more than the same
dependency on a twelve-row lookup. Still bounded -- each pair costs a probe
query and an `ANALYZE`. Recommendations went from 2 to 3, and both designed
traps are now among them.

### The measurement, alternated because one run of this settles nothing

Two identical "before" runs gave **14.9%** and **19.3%** headroom, which is
2.18's 18.6% run-to-run variation landing directly on the quantity being
compared. So: three rounds, alternating stats on and off, on the eight workload
queries whose filters touch the correlated columns.

| arm | native (median, range) | oracle | headroom |
|---|---|---|---|
| statistics **ON** | 595.9 [546.0, 635.3] | 567.4 [544.1, 634.1] | **0.4%** [0.2, 4.8] |
| statistics **OFF** | 602.0 [520.6, 651.7] | 481.4 [446.6, 530.6] | **18.6%** [14.2, 20.0] |

The headroom ranges are **disjoint**, which is this project's own standard for a
difference worth believing (2.32). Native is unchanged between the arms -- the
ranges overlap almost entirely -- so the mechanism is not "queries got faster".
It is that **PostgreSQL's own plan is now already the good one**, and the oracle
rises to meet it. There is nothing left for a plan-picker to find.

Three `CREATE STATISTICS` statements, one `ANALYZE`, and **98% of the
opportunity is gone**.

### What that means for this project, stated plainly

The queries in this measurement are the ones `data/schema.sql` was rebuilt to
contain (2.9): correlated columns that PostgreSQL mis-estimates because it
assumes independence. They are the project's best case, deliberately
constructed, and they are also precisely the case a `dependencies` statistics
object fixes outright.

So the honest ordering for anyone deploying this:

1. **Run the advisor and create the statistics.** It is three statements, it
   costs one `ANALYZE`, it helps every query against those columns whether or
   not this optimizer ever sees them, and on this workload it removes almost all
   of the headroom.
2. **Then run the optimizer on what remains.**

Doing it the other way round means a learned model spending its capacity
rediscovering, per query, an estimate the planner could have been told once.

### Why this does not make the project pointless

Three reasons, and the first is the one that matters.

**It does not generalise past the case it fixes.** Extended statistics correct
*single-table* multi-column selectivity. They do nothing for join-order errors,
which is the entire content of the Join Order Benchmark and where 2.55 measured
`17c` going from 21.6 s to 2.0 s. Nothing in `CREATE STATISTICS` reaches that.

**Someone has to know which columns.** The advisor is the thing that found them,
and it found them by measuring q-error against real probe queries rather than by
being told. That machinery is part of this project.

**The oracle ceiling is the honest metric, and it just moved.** A system whose
reported win shrinks when the database is fixed is a system whose metric is
measuring the right thing. If the headline had *not* moved, that would have been
evidence the oracle was measuring something other than real opportunity.

The uncomfortable version is still worth writing down: on the synthetic
workload, this project's central claim competes with three lines of SQL, and the
three lines win.

## 2.58 Buffers are a perfectly stable label for the wrong quantity

The roadmap's case for training on buffer counts instead of wall-clock latency:
latency varies **18.6%** between identical executions of the same plan (2.18),
and that noise goes straight into the label the model fits. Buffer counts are a
property of the plan and the data, not of what else the machine was doing.

Two things have to hold for the swap to be worth making, and only the first
does. Measured over 27,001 logged rows, every one of which already carries
`Shared Hit Blocks` because `get_plan` asks for `BUFFERS`:

### Buffers are stable. Enormously so.

Across 1,721 `(query, hint)` pairs executed three or more times:

| | median spread | mean spread |
|---|---|---|
| wall-clock latency | **96.4%** | 99.8% |
| buffer count | **0.0%** | 1.6% |

Buffers are **identical across every repetition** on 1,136 of 1,721 pairs. The
latency figure is larger than 2.18's 18.6% because this spans every run ever
logged rather than one sitting, which is the right comparison for a training
label: the model fits all of it.

So the premise holds completely. A label with zero variance where the current
one has ~96%.

### And they rank the candidates differently

| | value |
|---|---|
| queries with 4+ distinct candidates | 59 |
| Spearman(latency, buffers) within a query, median | **+0.341** |
| positive | 53 / 59 |
| **above +0.8** | **0 / 59** |

Right sign, nowhere near strong enough. Not one query in fifty-nine has buffers
ordering its candidates the way latency does.

That is not surprising once stated: two plans touching the same pages can differ
by a sort that spills, a hash that fits in `work_mem` or does not, a nested loop
re-reading cached pages cheaply. Buffers count *pages touched*; latency measures
*work done*. They are correlated because both grow with data volume, and they
diverge precisely on the operator choices this system exists to make.

### The verdict, and the useful thing underneath it

**Rejected as a replacement label.** A model trained on buffers would learn to
pick the plan that touches fewest pages, which is the right answer 34% as often
as the question is asked. That trades the current label's *noise* for *bias*, and
bias does not average out with more data.

**Worth keeping for two other jobs**, both of which follow from the stability
number rather than the correlation:

- **As a noise floor.** Two candidates with identical buffer counts did the same
  page work, so any latency difference between them is measurement noise. That
  is a per-pair noise estimate the project currently gets only from repeated
  execution, and it is free -- it is already in the log. 2.18 measures noise
  globally; this would measure it per comparison.
- ~~**As a feature.**~~ Checked before writing it down, and it does not work.
  Buffer counts come from `EXPLAIN ANALYZE`, and the production path plans every
  candidate with `analyze=False`, executing only the winner (2.6). Verified
  against a live planner: `get_plan(..., analyze=False)["buffers"]` is `None`.
  A feature populated in training and absent at serving is train/serve skew by
  construction -- the model would fit a signal guaranteed missing at the moment
  it matters. This is the same constraint that shapes the whole feature set, and
  it is worth noticing *before* adding the feature rather than after the
  deviation rate collapses.

The noise-floor use is open. What is closed is the version in the roadmap: the label stays
wall-clock, and the reason is measured rather than assumed.

## 2.59 The normalised bound was blamed for the wrong thing

`optimizer/conformal.py` defaults to an **absolute** bound -- `mean + q` rather
than `mean + q·sigma` -- and its docstring gives the reason: over 2,483
calibration rows the normalised version was unusable at every risk level worth
having, because rows where the eight ensemble members agree almost exactly give
a near-zero denominator that dominates the quantile.

2.46 took that at face value and proposed the obvious repair: the degenerate
denominators are the problem, not the normalisation, so floor sigma at a
percentile of the sigmas actually observed and the normalised bound should
become usable while keeping the per-candidate scaling the absolute one throws
away.

The floor is built -- `sigma_floor_percentile`, fitted at calibration time and
applied identically at serving, because conformal coverage requires the score
function to be the same in both places. Then it was measured, on 1,525 rows
across 59 queries with a query-level split:

| variant | q | held-out coverage | target |
|---|---|---|---|
| **absolute (current default)** | 0.2876 | **90.6%** | 90% |
| normalised, `MIN_SIGMA` only | 5.6073 | 83.8% | 90% |
| normalised, p25 sigma floor | 5.2700 | 83.8% | 90% |
| normalised, p50 sigma floor | 4.0825 | 85.8% | 90% |

The floor narrows the bound by 6% and moves coverage not at all. And the line
that explains why is the one above the table:

```
sigma: min 5.00e-03  p25 0.0186  median 0.0381  max 0.7512
rows with sigma < 1e-4: 0 (0.0%)
```

**There are no near-zero denominators.** The smallest sigma in the whole
calibration sample is 5e-3, five thousand times `MIN_SIGMA`. The failure the
docstring describes cannot be happening on this model, and flooring something
that never approaches the floor changes nothing.

### So why is normalised worse?

Because sigma is not proportional to error. 2.4.5 measured `uncertainty_scale`
at **3.75** -- the ensemble's spread understates its own error by nearly four
times -- and that mis-scaling is not uniform across rows. Dividing a residual by
a quantity that tracks error badly does not standardise the score, it adds a
second source of variation to it. The absolute bound never divides, so it never
inherits that.

Which makes the absolute default correct for a reason its own docstring does not
give. The stated reason -- degenerate denominators -- was true of whatever model
it was written against and is not true of this one; the real reason is the one in
2.4.5, and it does not go away with more data or a better floor.

### What was kept anyway

The floor stays, defaulted on for normalised gates and inert for absolute ones.
It costs nothing, it is the right guard for a model that *does* produce
degenerate sigmas, and a future ensemble with tighter agreement would hit
exactly the failure the docstring describes. Keeping a cheap guard against a
failure that has been observed once is not the same as acting on a diagnosis
that turned out to be wrong here.

`P6` is closed as measured-and-rejected. The conformal gate stays absolute, and
the reason recorded next to it is now the one the numbers support.

## 2.60 A feature that reads the question instead of the answer

Every feature in the vector is read off PostgreSQL's chosen plan: node types,
estimated rows, costs, tree shape. All of it describes what the planner
*decided* -- and the premise of this project is that what it decided is wrong in
exactly the cases with headroom. The plan is the answer; nothing described the
question.

The specific gap is 2.9's mechanism. PostgreSQL estimates `WHERE a = x AND b =
y` as sel(a) x sel(b), which is right only when the columns are unrelated, and
`data/schema.sql` was rebuilt to contain exactly that violation. A plan built on
that assumption cannot reveal it: the row estimate it carries is the assumption's
output.

### Measured before building it

The hypothesis is testable without writing a feature: if same-table equality
conjunctions are where the estimate goes wrong, they should be where the
headroom is. Across 59 workload queries, median available headroom:

| same-table equalities | queries | median headroom | mean |
|---|---|---|---|
| 0 | 30 | 16.9% | 19.2% |
| 1 | 3 | 25.3% | 30.5% |
| **2 or more** | 26 | **31.4%** | 35.2% |

Monotone, nearly double at the top, and it agrees with the hand-written `trap`
labels without having been shown them.

### What was added

`features.predicate_shape()` reads the WHERE clause for three numbers:
`max_same_table_equalities`, `n_equality_predicates`, `n_range_predicates`.

Join keys are deliberately excluded. `o.user_id = u.id` is an equality, and it
estimates a *join*, not a selectivity on one table -- counting it would put the
largest number on the widest query rather than the most mis-estimated one. The
largest count on any single table is the feature, not the total, because two
filters on two tables are never multiplied together and do not exercise the
assumption at all.

### The part that had to be got right

`sql_text` now travels on the candidate dict, attached in **three** places:
`train._row_to_candidate`, `planner.plan_query` and `benchmark.run`. A test
asserts all three.

That is not defensive habit. A feature populated during training and absent at
serving is train/serve skew by construction -- the model fits a signal that is
guaranteed missing at the moment it is used -- and this session alone found that
shape four times: the policy (2.43), the candidate pool (2.47), the swept
`gate_mode` (2.50), and 39% of the action space classified as `other` (2.56). It
is also why buffer counts were *rejected* as features two sections ago: they come
from `EXPLAIN ANALYZE`, and the serving path plans with `analyze=False`.

Query text is the opposite case. It is available before anything executes, on
every path, which is what makes it usable at all.

### Not yet worth anything

The feature exists and no model has been trained with it. Availability is not
usefulness, and this project has a measurement to prove it: `num_joins` has been
in the vector the whole time and gets **zero** splits across all eight ensemble
members. The honest status is that the signal is in the data and the model has
not yet been given a chance to use it.

## 2.61 pg_overexplain is nearly free and has nothing to sell

PostgreSQL 18 ships `pg_overexplain`, which exposes planner internals the plan
JSON does not. The roadmap listed it as a feature source. Two measurements
settled it, in the order that costs least.

**It is nearly free.** Over 320 EXPLAINs on the wide workload queries:

| option | median | vs plain |
|---|---|---|
| `FORMAT JSON` | 0.620 ms | — |
| `+ DEBUG` | 0.625 ms | **+0.8%** |
| `+ RANGE_TABLE` | 0.621 ms | +0.1% |

Which removes the obvious objection -- that a serving path planning up to 96
candidates cannot afford a richer EXPLAIN. It can.

**And three of its four candidate fields do not vary.** A field constant across
the candidates for one query cannot separate them, whatever it correlates with
across queries. That is the `has_hint` lesson from the feature audit: always
1.0, and in the vector for months.

| field | varies within a query |
|---|---|
| `Parallel Safe` | **56 / 59** |
| `Subplans Needing Rewind` | 0 / 59 |
| `Disabled Nodes` | 0 / 59 |
| `Lateral` | 0 / 59 |

`Disabled Nodes` reads zero because `plan_query` already filters those
candidates out (2.29); `Lateral` because this workload has none. Both are
correct zeros rather than missing data.

**The one that varies does not predict.** Spearman between the parallel-safe
node count and measured latency, within each query:

| | value |
|---|---|
| median | **+0.040** |
| positive | 32 / 56 |
| \|rho\| > 0.5 | 15 / 56 |

A median of +0.04 on 56 queries is a coin flip, and the strong cases point both
ways -- `shape_window_rank` at **+0.925** and `gen_15_4w` at **-0.732**. It is
picking up something query-specific rather than a general property of a faster
plan.

Rejected. Not because it is expensive -- it costs 0.8% -- but because the
measurement says there is nothing there. Worth recording precisely so the next
reading of the PG18 release notes does not spend the same afternoon.

The contrast with 2.60 is the useful part. Both are "add features"; one was
tested by asking whether the quantity tracks headroom **before** any code was
written and came back monotone at 16.9% / 25.3% / 31.4%, and the other came back
at +0.04. The cost of asking first was one script each.

## 2.62 One plan for a whole template: already done, and checked for safety

The roadmap asked for parameterised queries to share a plan decision:
`GROUP_BY_TEMPLATE` groups them for *identity*, and nothing was said to exploit
the fact that one plan may serve them all.

It already does. `query_fingerprint` applies `query_template()`, so
`city = 'Mumbai'` and `city = 'Singapore'` hash to the same id, and
`decision_cache` is keyed on that id. Choose once, serve the template. Verified
directly: the two produce `adhoc:e0e00d23...` identically.

Which leaves the question the entry did not ask, and it is the more interesting
one. **Should** they share? Two literals with different selectivity can want
different plans, and a cache handing the Mumbai decision to a Singapore query is
serving a plan chosen for a filter seven times larger.

### Measured, on a template where hints actually win

`corr_4w_full_correlated`'s shape -- 63.2% headroom, so "which hint wins" is a
real question. Every servable candidate executed, per literal:

| literal | matching rows | native | best | headroom | winning hint |
|---|---|---|---|---|---|
| Mumbai | 15,151 | 140.0 | 93.7 | 33.1% | `Set(enable_nestloop off)` |
| Delhi | 6,267 | 121.8 | 80.7 | 33.7% | `Set(enable_nestloop off)` |
| Bangalore | 4,873 | 128.7 | 85.5 | 33.6% | **`Leading(o u oi p)`** |
| Singapore | 2,245 | 120.6 | 74.2 | 38.5% | `Set(enable_nestloop off)` |

Two distinct winners across a 7x selectivity span, which looks like an argument
against sharing. Then the cost of actually sharing the first literal's plan with
all four:

| literal | own best | served the shared plan | penalty |
|---|---|---|---|
| Mumbai | 93.7 | 96.7 | +3.2% |
| Delhi | 80.7 | 86.9 | +7.6% |
| Bangalore | 85.5 | 74.0 | **-13.5%** |
| Singapore | 74.2 | 79.3 | +6.8% |

Median +5.0%, worst +7.6% -- and Bangalore, the literal with a *different*
winning hint, comes out **13.5% faster** on the shared plan than on its own
supposed best.

That last number is the answer. A plan measured as worse cannot be 13.5% faster
than the winner; what it means is that the difference between these plans is
below the measurement floor. 2.18 measured 18.6% variation between identical
executions, and every figure in that second table is inside it.

### So: safe here, and the caveat is specific

Sharing one decision across this template costs nothing measurable. The two
"different winners" are two hints landing on indistinguishable plans, which is
what 56 of 83 candidates collapsing to duplicates (`plan_query`'s budget
comment) would predict.

What this does *not* establish is safety at wider selectivity spans. Seven times
is the whole range this schema offers on `city`; a production workload where one
literal matches 10 rows and another matches 10 million is a different question,
and the honest answer there is that nothing here has tested it. The mechanism
that would break -- a plan chosen for a large filter being served to a tiny one
-- is real even though it does not fire at this scale.

## 2.63 The wider action space raises the ceiling and the risk together

2.56 widened the search: the join-order sample scales with width, scan methods
are a swept family, and collection mirrors serving. Re-collecting the synthetic
workload under it produced 4,499 rows across 65 queries, and training on them
gave the best train-time numbers this project has recorded:

| | champion | challenger |
|---|---|---|
| typical multiplicative error | 1.31x | **1.27x** |
| `pairwise_rank` headroom captured | 62.7% | **78.8%** |
| features | 64 | 67 |
| held-out queries | 15 | 16 |

And `app.retrain` **rejected** it:

```
challenger's worst pick is 3.09x the best available plan against the
champion's 1.59x -- a better mean does not buy a worse worst case
```

### The yardstick was clean, which makes the verdict worth reading

2.45's fix earned its keep here. The comparison ran on six queries **neither
model trained on** -- `yardstick_provenance` reports `champion_trained_on: 0,
challenger_trained_on: 0` -- which is the first genuinely uncontaminated
promotion comparison in the project's history.

On it, the challenger is better on two of the three worst cases (1.39x against
1.59x, 1.22x against 1.40x) and catastrophic on one.

### What happened on that one query

`shape_group_by_category`, 293 candidates in the widened space, native 141.4 ms,
best available **55.0 ms**:

| | picks | ran at | predicted |
|---|---|---|---|
| champion | `SeqScan(o)` | **57.3 ms** | 0.832x |
| challenger | `Set(enable_nestloop off)` | **170.1 ms** | **0.610x** |
| best available | `SeqScan(o)` | 55.0 ms | — |

The champion, which never trained on a single scan-method candidate, picks the
scan-method winner. The challenger, which trained on thousands of them, picks an
operator toggle that runs slower than native.

**And the safety gate cannot catch it.** The challenger predicted **0.610x** for
a plan that ran at 1.20x native. The gate reads the prediction; a confident
wrong prediction is exactly the input it has no defence against. This is
2.4.5's finding arriving where it hurts: the ensemble's spread understates its
error, so a badly wrong point estimate does not come with a wide enough interval
to be declined.

### The general shape

A wider action space is not a free improvement. On this query it raised the
ceiling from native's 141.4 ms to 55.0 ms -- **61% available**, far more than the
narrow space offered -- and simultaneously introduced a 170 ms plan that the
model rates better than the 55 ms one.

More options means more upside and more ways to be wrong, and which dominates is
a property of the model's accuracy rather than of the action space. At 1.27x
typical multiplicative error the model can exploit part of the new ceiling and
not all of it, and the promotion rule is what stops the difference being served.

The rejection is the system working. It is also a bill: the ceiling moved and
the model cannot yet reach it safely, which is a better problem than the one in
2.51 (nothing to reach) and is still a problem.

## 2.64 A model trained on six tables ranks plans on twenty-one

> **Superseded by 2.86 — and not only the levels.** Re-run on the wide action
> space, the carried model captures **−5.3%**: it serves JOB 4% *slower* than
> doing nothing, against a local model's 96.5%. The comparison this section is
> built on does not survive. What does: the carried model still beats random
> selection by 22 points, so the structural features carry real signal — just
> not enough to clear native.

`README` claims this works on any PostgreSQL database, and the evidence has
always been that the *pipeline* runs anywhere -- schema discovery, workload
generation, training. Whether a trained model carries across schemas was never
tested.

The feature vector is half schema-specific: one `present` / `join_position` /
`selectivity` / `index_scan` / `occurrences` slot per table. Carried to a
different database those slots read **zero**, because the table names do not
match. So this measures exactly one thing: is there enough in the
schema-agnostic half -- costs, tree shape, join-method counts, and 2.60's
predicate shape -- to rank plans on a database the model has never seen?

Both directions, with **zero** overlapping tables in each:

### Synthetic model (6 tables, 755k rows) carried to JOB/IMDB (21 tables, 73M rows)

| selector | total | of native | headroom captured |
|---|---|---|---|
| native | 116,436 ms | 100.0% | — |
| cost argmin | 105,389 ms | 90.5% | 12.8% |
| random pick | 146,958 ms | 126.2% | **-35.4%** |
| **carried synthetic model** | **32,732 ms** | **28.1%** | **97.0%** |
| locally-trained JOB model | 31,508 ms | 27.1% | 98.4% |

**Within 1.4 points of the model trained on that database.**

### JOB model carried to the synthetic schema

| selector | total | of native | headroom captured |
|---|---|---|---|
| native | 10,711 ms | 100.0% | — |
| cost argmin | 10,242 ms | 95.6% | 12.1% |
| random pick | 15,772 ms | 147.3% | -130.7% |
| **carried JOB model** | **9,003 ms** | **84.1%** | **44.1%** |
| locally-trained model | 8,025 ms | 74.9% | 69.3% |

Weaker -- 44.1% against 69.3% -- and still four times the cost model's 12.1%.

### What it does and does not say

**It transfers, and the direction matters.** Carrying *to* JOB works almost
perfectly and carrying *from* it works partially. JOB's headroom is enormous and
concentrated -- native 116 seconds against an oracle of 30 -- so a selector only
has to avoid the disasters to capture most of it by latency, and generic
structural features are enough for that. The synthetic schema's headroom is
spread thinner, and picking well there needs more.

**Total latency captured is not accuracy.** The carried synthetic model picks
the exact best candidate on **10 of 30** JOB queries, and the carried JOB model
on **5 of 65** synthetic ones. It is good at avoiding bad plans and mediocre at
finding the best, which is 2.43's finding about the ensemble arriving by a
different route.

**The per-table features may be doing less work than their share of the vector
suggests.** A quarter of the vector is per-table slots, they read zero in both
experiments, and one of these transfers landed within 1.4 points of a local
model. That is consistent with the feature audit, which found all six
`*_occurrences` slots among the sixteen features **never split on** by any
ensemble member.

The claim "works on any PostgreSQL database" now has a measurement behind it
rather than an architecture diagram -- and a specific caveat: it is the
structural half of the model that travels, and how far it gets depends on
whether the target's headroom is concentrated enough to reach without local
knowledge.

## 2.65 Running the benchmark blinds the sweep that tunes it

`app.autotune` sweeps the selection policy against the action space against the
gate, and the point of the middle axis is to answer the question 2.27 left open:
do `resource`, `join_method` and `scan_method` earn their place, or are they off
because someone switched them off in 2024?

The first run came back with the axis completely flat. Not "close" -- identical,
every combination of optional families reporting the same net, the same
deviation rate, the same regression rate and the same worst case:

```
        policy           extra families     z  gain  dev%  reg%   worst    net%
 pairwise_rank                     none  0.50  0.05   53%   25%   1.00x   11.5%
 pairwise_rank                 resource  0.50  0.05   53%   25%   1.00x   11.5%
 pairwise_rank              join_method  0.50  0.05   53%   25%   1.00x   11.5%
 pairwise_rank              scan_method  0.50  0.05   53%   25%   1.00x   11.5%
```

An axis that returns the same answer for every setting is either a finding or a
blind spot, and the two look identical from the outside. "Is this code reached?"
is the question that separates them, so: count what the sweep is actually
scoring.

### 117 candidates, three families

The replay was scoring **117** candidates across the held-out queries --
`operator_toggle` 69, `join_order` 43, `cardinality` 5. Zero `resource`, zero
`scan_method`, zero `join_method`. The three families being priced were not in
the data. The sweep was not measuring them as worthless; it was measuring their
absence and reporting it as a tie.

The log has them. `corr_5w_premium_supplier` alone carries **976** candidate
rows across **50** runs. What it does not have is them in the run the replay
chose.

### Two kinds of run write to the same table

```
 selector_used |          ts            |  n
 learned       | 2026-09-05 07:20:22+00 |  12   <- picked
 learned       | 2026-09-05 07:05:13+00 |  12
 learned       | 2026-09-05 06:49:56+00 |  12
 collection    | 2026-09-05 06:12:23+00 |  84
```

`app.collect_data` **explores**: it walks the action space and writes every
candidate it executed, all eight families. `app.benchmark` **serves**: one
decision per query, a dozen rows, and only from the families the current
deployment has switched on.

`_aggregate_repetitions(latest_run_only=True)` exists for a good reason -- 2.32
measured the same native query ranging from 2.4 ms to 13.7 ms across runs
collected months apart, so a *comparison* needs one run's conditions rather than
a decade's. It implemented that as "the newest `created_at` per query". And a
benchmark run is always newer than the collection that preceded it, because you
collect and then you benchmark.

So the filter reliably picked the narrowest run in the log. Across the queries
being scored that is **117 of 513** candidates -- **77% of the action space
hidden from the tool whose job is to choose it** -- and on
`corr_5w_premium_supplier`, 12 rows where the collection had 84.

**Running the benchmark blinds the sweep that tunes the benchmark.**

### The fix, and the fix being inert

Prefer the newest *collection* run per query; fall back to the newest run of any
kind only for a query nothing has ever collected. That keeps the original
purpose intact and restores the action space.

Written, and it changed nothing. `FETCH_SQL` did not select `selector_used`, so
`row.get("selector_used")` was `None` on every row, never equal to
`COLLECTION_SELECTOR`, and every query fell through to the fallback -- which is
the old behaviour exactly. A fix whose only symptom is a plausible number,
inside the fix for a bug whose only symptom was a plausible number.

Adding the column is one line. A regression test asserting the column stays
selected is the part worth keeping, because nothing else in the system fails if
it disappears again.

| | before | after |
|---|---|---|
| candidates scored | 117 | **513** |
| `operator_toggle` | 69 | 161 |
| `join_order` | 43 | 128 |
| `scan_method` | 0 | **119** |
| `resource` | 0 | **105** |
| `cardinality` | 5 | 0 |

### What the axis says now that it can see

With the families visible, the widened space changes the ranker's pick on
**5 of 15** held-out queries -- so the axis moves. The numbers it produces are
still nearly identical, and now that is a result rather than an artefact:

| space | ranker's pick changes | net served |
|---|---|---|
| narrow | — | +5.27%, 73.8 ms |
| `+resource` | 1 query, by 1.0 ms | +5.27%, **73.8 ms** |
| `+scan_method` | 4 queries | +4.77%, 66.8 ms |

`+resource` is identical to the tenth of a millisecond, not merely to the
decimal place: the one pick it changes is a deviation the gate declines, so the
family is visible to the ranker and invisible in what gets served.

`resource` is the oracle on 3 of the 14 held-out queries with any headroom and
`scan_method` on 2, so the families do contain the wins. The ranker does not
find them: it changes its pick often enough to prove the space is live, and the
changes are a coin flip. That is 2.43's finding again -- the constraint is the
comparison, not the menu.

The honest reading of the action-space axis is therefore **"no evidence either
way at this resolution"**, and 2.32's 8-point floor says a 0.5-point difference
is not evidence. The defaults stay where they are because nothing has beaten
them, which is a different sentence from "they were measured best".

### The pattern

This is the third instrument in this project found measuring a narrower world
than the one being served -- after the calibration replay scoring an action
space no query is offered (2.47) and 39% of the space classifying as `other` and
being dropped (2.56). All three had the same signature: a defensible filter,
written for a real reason, silently doing more than its name says.

And this one has a feedback loop the others did not. The benchmark is what tells
you whether the configuration is any good; the sweep is what chooses the
configuration; and running the first was corrupting the second. The more you
measured, the less the tuner could see.

## 2.66 The held-out number is mostly the split

2.59 noticed the conformal bound achieving **90.6%** coverage on one query-level
split and **75.5%** on `app.train`'s own held-out list, and filed it as an
oddity. `app.coverage_sweep` turns it into a measurement: ten query-level
splits, the same rows, the same model class, one integer different.

```
  seed  held-out qs  test rows   coverage  (raw means)  typical err   captured
    42           16        562      71.7%        72.8%        1.30x      48.5%
    43           16        568      86.3%        87.1%        1.45x      89.1%
    44           16        578      62.5%        62.8%        1.37x      30.9%
    45           16        571      78.5%        78.3%        1.33x      60.1%
    46           16        557      89.4%        88.9%        1.31x      78.7%
    47           16        605      71.2%        71.4%        1.31x      45.3%
    48           16        580      77.2%        78.1%        1.38x      82.6%
    49           16        571      92.6%        92.5%        1.41x      24.2%
    50           16        550      73.8%        73.1%        1.25x      28.6%
    51           16        596      73.3%        72.7%        1.31x      61.9%
```

### The guarantee does not hold

Coverage: median **75.5%**, range **62.5% to 92.6%**, and it reaches the
advertised 90% on **one split in ten**.

This is not a bug in the conformal implementation. Split conformal promises
1 - alpha coverage, distribution-free, **under exchangeability of the
calibration and test rows** -- and a query-level split is built to break
exchangeability. Rows from one query share a plan shape, a table set, a native
latency and a headroom; holding out whole queries means the test rows come from
queries the calibration rows have never described. The theorem does not apply,
and the measurement says how much that costs: about **15 points of coverage in
the median, and up to 27 in the tail**.

The honest way to report it is the range. The bound remains useful as a
calibrated-*ish* interval fitted on real residuals rather than a chosen `z`, and
it is not the 90% it advertises. `docs/METRICS.md` and the gate's own
documentation now say so.

### The bigger number is the one nobody was watching

Headroom captured, same ten splits, same everything: **24.2% to 89.1%**. A
**64.9-point** range.

That is the project's headline train-time metric, and one split is one draw from
that. 2.63 reads a champion at 62.7% against a challenger at 78.8% and treats
the 16-point gap as informative; both were measured on seed 42, and 16 points is
a quarter of the spread the seed alone produces. The promotion decision in 2.63
does not rest on it -- `app.retrain` compares on worst-case regression over a
common held-out set, which is why it rejected a model with the better mean --
but the sentence around it was doing more work than the number could support.

2.32 measured the *benchmark* unable to resolve differences under about 8
points. This says the *training* report cannot resolve differences under
something far larger, and for a different reason: the benchmark's problem is
measurement noise, and this is sampling. Sixteen queries is a small sample of a
65-query workload, and the queries differ enormously in how much headroom they
carry.

### Two things it is not

**Not nondeterminism.** Checked, because an earlier pair of sweeps disagreed on
two of ten seeds. Training is bit-identical across three runs in one process,
three separate processes on a quiet machine, and three more with eight CPU-bound
processes competing -- twelve runs, one number. `pg_class.reltuples`, which is
the denominator of every selectivity feature and is re-`ANALYZE`d on each run,
is also stable: five ANALYZEs, six tables, zero movement. The original
disagreement remains unexplained and did not recur.

**Not the calibration skew, though that was real.** The coverage check fed the
gate **raw** ensemble means. The bound is *fitted* on point-calibrated means --
the comment at the fit site says fitting on raw means and shifting the point
underneath would invalidate the guarantee -- and `learned._predict` applies the
same correction before the gate reads it. So the one number that says whether
the gate is honest was measured against a gate nothing consults. Fixed, both
figures now reported, and the difference is **under one point** on every split.
A train/serve skew that turned out not to matter is still worth the ten minutes:
the alternative was not knowing.

### What to do about it

Report the range. `app.coverage_sweep` costs one retrain per seed -- about two
minutes for ten on this workload -- which is cheap enough that there is no
excuse for quoting a single split.

The deeper fix is more queries. Sixteen held out of 65 is the sample size doing
the damage, and the same fix answers P7's selection-effect problem. Everything
else here is a way of being honest about a small sample rather than a way of
making it larger.

## 2.67 Pricing the optional hint families, on both databases

With 2.65's fix the action-space axis can see what it is pricing, so the
question 2.27 left open -- do `resource`, `join_method` and `scan_method` earn
their place, or are they off because someone switched them off? -- is one
command on each database.

### Synthetic, 15 held-out queries

| policy | extra families | net | deviates | regresses | worst |
|---|---|---|---|---|---|
| `pairwise_rank` | none | **+5.27%** | 53% | 25% | 1.00x |
| `pairwise_rank` | `resource` | +5.27% | 53% | 25% | 1.00x |
| `pairwise_rank` | `join_method` | +5.27% | 53% | 25% | 1.00x |
| `pairwise_rank` | `scan_method` | +4.77% | 53% | 25% | 1.00x |
| `greedy` | none | +1.39% | 27% | 25% | 1.00x |
| `risk_averse` | `scan_method` | +0.16% | 27% | 25% | 1.00x |

The policy axis is decisive and the action-space axis is not. `pairwise_rank`
is nearly **four times** `greedy` and thirty times `risk_averse`, which is
2.43's result reproduced on a wider action space and a bigger workload.

Adding `resource` moves the served total by **0.0 ms**. Adding `scan_method`
costs **0.50 points**.

**The `join_method` row is not a measurement**, and 2.75 is why: the log holds
no join-method rows, so that arm was a copy of the `none` arm and the tie was
arithmetic rather than evidence. The sweep now says which families it could not
score instead of scoring them as worthless.

### JOB, 8 held-out queries

Every one of the 32 configurations qualifies, and they land between **87.22%**
and **87.23%**. The whole sweep spans **0.01 points**.

JOB's headroom is enormous and concentrated -- 37 seconds of net improvement
across 8 queries -- so the entire task is avoiding catastrophes, and every
policy and every action space avoids them. The benchmark that made the case for
learned optimizers cannot distinguish between the configurations of one.

That is worth stating plainly, because JOB is the standard evidence and it is
being asked a question it cannot answer. The synthetic workload, which is
smaller and less impressive, is the one with resolution here.

### The incumbent rule fired, correctly

On JOB, `greedy` scored 87.23% against `pairwise_rank`'s 87.22% -- and the
report says `kept_incumbent_over: greedy`. `MIN_POLICY_SWITCH_POINTS` refused to
recommend switching the served policy on a **0.01-point** difference.

That guard exists because this tool once recommended exactly such a switch on a
0.8-point difference, having printed two lines earlier that a difference that
size was inside its own spread (2.50). Applying it would have re-created 2.43 --
a gate calibrated under one policy while the server selects with another --
using the instrument built to prevent it.

### The verdict

**The defaults do not change.** `SELECTION_POLICY=pairwise_rank`, and
`INCLUDE_TUNING`, `INCLUDE_JOIN_METHODS` and `INCLUDE_SCAN_METHODS` all off.

What changed is the reason. They were off because a 24-query measurement in 2.27
said so, before the workload tripled and before PostgreSQL 18. They are off now
because a sweep that can see them, on two databases, found no configuration that
beats them -- and, on the synthetic workload, one that loses by half a point.

The correct reading of the action-space axis is **no evidence either way at this
resolution**, not "measured worthless". The families do contain wins: across the
held-out queries `resource` is the single best candidate on 3 of 14 and
`scan_method` on 2, and widening the space changes the ranker's pick on 5 of 15
queries. The ranker changes its mind often enough to prove the space is live,
and the changes are a coin flip. The binding constraint is the comparison, not
the menu -- which is P4, and is where the next real gain is.

## 2.68 Attributing a seven-point drop, and not believing the first answer

Adding six unfiltered six-table queries moved the curated benchmark figure from
**33.0%** to **26.0%**. The open question was whether that is dilution -- new
queries with nothing to win, which are arithmetically neutral -- or a
regression, the model deviating and losing on them.

### The premise was wrong first

The six were described as "zero-headroom queries", on the grounds that every
earlier batch was screened by a >=8% headroom filter and these were not.
Measured, they carry **230.5 ms of headroom on 977.8 ms of native -- 23.6%**,
which is *more* than the workload average. Unfiltered is not the same as
headroom-free; the filter selected for headroom, and its absence lets in queries
with a lot as easily as queries with none.

So the neutrality argument does not apply, and the drop has to be attributed
rather than dismissed.

### What one deploy pass said

| query | native | served | outcome |
|---|---|---|---|
| `corr_6w_supplier_country` | 467.1 | 467.1 | safety veto |
| `shape_6w_distinct_pairs` | 97.9 | 97.9 | guard blocked |
| `ctl_6w_uncorrelated` | 75.5 | 75.5 | guard blocked |
| `shape_6w_anti_join` | 114.1 | 114.1 | safety veto |
| `corr_6w_band_category` | 95.8 | **104.2** | deviated, lost |
| `shape_6w_top_suppliers` | 127.4 | **162.2** | deviated, lost |

Captured **-18.7%** of the available headroom. Four of six never deviate at all:
two safety vetoes and two guard blocks. The protections carry the group, and
where the model is allowed to act it is 0 for 2.

### And what seven repetitions said

A single pass is not evidence -- 2.18 measured repeated executions of the same
plan varying 18.6%, and this project has published a single-pass number before.
Paired and alternated, native and hint in turn so that 2.32's between-run drift
hits both arms:

| query | native | hinted | one pass said | repeated says |
|---|---|---|---|---|
| `corr_6w_band_category` | 98.6 | 100.3 | 1.087x | **1.017x** |
| `shape_6w_top_suppliers` | 120.0 | 131.4 | 1.273x | **1.095x** |

Neither range is disjoint from its native range. The losses are real in
direction and roughly a third of the size the deploy pass reported: about
**13 ms** across the group rather than 43, which puts the captured figure near
**-5.7%** rather than -18.7%.

The oracle in that table is a single-pass minimum too, and a minimum over noisy
draws is biased low -- so the denominator is generous and the true figure is
closer to zero still.

### The answer

**It is a regression, and it is small.** The model does deviate and lose on this
group, so the drop is not pure dilution; and the effect is a few milliseconds
against a 7-point move that 2.32's 8-point floor and 2.66's split range both
swallow whole. The right reading of 33.0% -> 26.0% is that it is not a
measurement of anything.

### One deviation exceeded its own bound; one did not

| query | pessimistic bound | actually ran at | inside? |
|---|---|---|---|
| `corr_6w_band_category` | 0.92x | 1.017x | **no** |
| `shape_6w_top_suppliers` | 1.20x | 1.095x | yes |

One in two, on a bound advertised at 90% -- consistent with 2.66 measuring its
real coverage at 75%, and a reminder that the coverage number is not an
abstraction: it is the rate at which the gate's promise about a served plan
turns out to be false.

The second row is worth reading carefully, because it looks like a gate failure
and is not. `shape_6w_top_suppliers` was authorised with a pessimistic estimate
of **1.20x native** -- the gate knowingly permitted a plan it believed was
probably slower. That is the rule as designed: the second condition asks whether
the downside is *bounded* (below 1.25x), not whether the worst case is a win,
and 2.18 records what happened the last time that was tightened -- a 96% veto
rate and a system that never deviated at all. Here the design was right and the
plan came in under its bound.

### What this does and does not license

It does not license turning the guard down. Four of six deviations were stopped
and the two that got through lost; that ratio is the guard working on a
workload the model handles badly.

It does not implicate the action space either. `corr_6w_band_category`'s pick
was an operator toggle and `shape_6w_top_suppliers`' was a cardinality
correction -- both from the always-on families, neither from anything 2.67
priced.

What it implicates is the ranking. On six queries with 23.6% headroom available,
the model's two authorised choices were both worse than doing nothing, and both
were queries it had **trained on**. That is P3, and it is where the remaining
value is.

## 2.69 A third of the feature vector cannot separate anything, and it barely matters

2.60 added three query-shape features and did the right thing first: it measured
`max_same_table_equalities` monotone against headroom -- **16.9% -> 25.3% ->
31.4%** across its three levels -- before a line was written. That is the
"is this worth building?" discipline this project keeps crediting.

The features are in the challenger bundle and they are used: all eight ensemble
members split on all three, **557 splits, 3.0%** of the total. So the natural
follow-up -- "do the splits land on them?" -- comes back yes.

### Then ablate them

Same rows, same seed, everything else identical, `--exclude-feature` for each:

```
with      features=67   captured=0.4647327672   err=1.299495
without   features=64   captured=0.4647327672   err=1.301953
```

Two different rankers, trained on different vectors, and the held-out capture
agrees to **ten decimal places**. Checked directly: the pick is the same hint on
**16 of 16** held-out queries.

### Why: the monotonicity was measured on the wrong axis

`pairwise_rank` compares candidates **within one query**. These three features
are functions of the SQL text, and every candidate for a query has the same SQL
text -- a hint changes the plan, not the predicates. So all three take one value
per query, and a feature constant across the things being compared cannot
contribute to the comparison.

2.60's measurement was real and it answered a different question. It showed the
feature predicts **which queries have headroom**, and the ranker is never asked
that. It is asked which plan, for this query, is faster.

This is the third instance:

| | what it was | found by |
|---|---|---|
| `has_hint` | always 1.0, in the vector for months | 2.50 |
| `Parallel Safe` | constant across a query's candidates | 2.61 |
| the three shape features | one value per query, by construction | here |

2.61 got it right and wrote the rule down -- "a field constant across a query's
candidates cannot separate them, whatever it correlates with across queries" --
in a scratch probe that was deleted. It is now `app.feature_audit`.

### The audit found twenty-three

```
models/_challenger.pkl: 67 features, over 65 queries with 2+ servable candidates

constant across every query's candidates: 23/67
splits spent on them: 1,250 of 18,290 (6.8% of the model's capacity)
```

Every `*_present` and `*_occurrences` slot, `num_joins`, `num_tables`,
`tree_leaves`, `n_disabled_nodes`, `log_root_width`, and `has_hint` -- which
2.50 identified as useless and which is still in the vector, still collecting
**255 splits**.

### On JOB it is fifty-two percent

The same audit against the 21-table schema:

```
models/plan_selector_job18.pkl: 139 features, over 30 queries
constant across every query's candidates: 72/139
splits spent on them: 421 of 12,902 (3.3% of the model's capacity)
```

More than half the vector, and the effect scales with schema width: 21 tables
means 105 per-table slots, and a query that touches six of them leaves 15 slots
reading the same value for every candidate it has.

The wasted *capacity* is smaller than the synthetic model's 6.8%, and for a
reason worth stating: a slot that is zero on every row of the whole dataset is
constant, useless and also un-split-on, so it costs nothing but a multiply. The
expensive ones are features that vary **across** queries and not **within** them
— `log_root_width` alone takes 247 splits on JOB and 289 on the synthetic
schema. Those are what a tree will happily spend itself on.

### Dropping all twenty-three makes the model better

Paired across sixteen query-level splits, because 2.66 measured a single split
worth up to 64.9 points on its own:

| | with all 67 | with 44 |
|---|---|---|
| better on | 5/16 splits | **11/16 splits** |
| mean held-out capture | — | **+8.3 points** |
| median | — | +1.4 points |
| best / worst split | — | +42.5 / -3.7 |
| typical multiplicative error better on | 3/16 | **13/16** |

Signed-rank on the capture deltas: **z = +2.22, p = 0.026**. The sign test alone
gives p = 0.210, and the difference between the two is eleven gains averaging
**+12.8** points against five losses averaging **-1.5**.

### And the attribution above is wrong

That paragraph says "dropping twenty-three features helps", and the natural
reading -- the one this section was originally written with -- is that the
*class* is the problem. It is not. Isolating the arms says so plainly:

| arm | what it drops | mean vs base | identical to base on |
|---|---|---|---|
| `hint` | `has_hint` alone | **+8.3** | 0 of 16 splits |
| `rest` | the other 22 | **+0.0** | **16 of 16 splits** |
| `all` | all 23 | +8.3 | 0 of 16 |

Dropping twenty-two of the twenty-three changes **nothing**, on every split, to
the last decimal. One feature is the entire effect, and 2.70 is what it turned
out to be.

So the class finding survives only in its weak form: 23 of 67 features cannot
separate a query's candidates, they are dead weight, and removing them is free.
Free is not the same as valuable, and this section originally claimed the
latter.

The audit is still worth having -- it is what found `has_hint` -- and the
23-of-67 and 72-of-139 counts are real. What is not real is the implied causal
story, and the cheap way to have caught it was the arm that was not run:
**ablate the class minus the suspect**.

## 2.70 One feature knew whether it was looking at PostgreSQL's own plan

2.69 found twenty-three features constant across a query's candidates and
measured dropping them worth a mean +8.3 points. Isolating the arms showed
twenty-two of them worth exactly nothing -- identical results on 16 of 16
splits -- and `has_hint` worth all of it.

`has_hint` is 1.0 when a candidate carries a hint. Every candidate carries a
hint. It has been in the vector since the beginning and 2.50 already identified
it as useless, and it was still there, still collecting **255 splits**. This is
why it survived.

### The training rows contain a row type that serving does not

`app.train` trains on **2,322** rows, of which **65** are baselines -- the
native plan for each query, `hint = NULL`, `has_hint = 0.0`. They are the only
rows where it is zero.

The pairwise ranker forms pairs within a query, so those 65 rows become
**2,257 of 42,782 pairs (5.3%)**, each one "is this hinted candidate faster than
native?". And `has_hint` answers half of that question perfectly.

At serving, `learned.select()` is handed candidates only. The baseline is the
gate's business -- selection and authorisation are separate steps -- so the
ranker is never asked that comparison, and in the population it *is* asked
about, `has_hint` is constant at 1.0.

A feature that is informative in training and constant in serving is not a dead
feature. It is a **train/serve skew wearing a dead feature's clothes**: 255
splits that separate training data along an axis that does not exist in
production. 2.47 found the same baseline-is-not-a-candidate confusion in the
calibration replay and in `retrain._score_bundle`. This is the third instance
and the first one that was costing accuracy rather than credibility.

### The obvious fix is wrong

If baseline pairs are the problem, remove them. Sixteen paired splits:

| | mean vs base | better on |
|---|---|---|
| drop the `has_hint` **feature**, keep the pairs | **+8.3** | 11/16 |
| drop the baseline **pairs**, keep every feature | **-4.2** | 6/16 |

Removing the pairs *loses* four points. Best guess, and it is a guess: the
baseline is the only row in each group whose ratio is exactly 1.0, so it anchors
where "no better than native" sits. Take it away and the ranker orders
candidates against each other with nothing to say whether any of them is worth
serving.

`rank_on_candidates_only` exists, is off, and its docstring carries this table.
The argument for it was good; the measurement was against it.

### The tool built to find this could not find it

`app.feature_audit` computes constancy over servable candidates and flags
`has_hint` correctly. `app.train --drop-constant-features`, written an hour
later, computed it over *training rows* -- baselines included -- so it dropped
the twenty-two that do not matter and kept the one that does.

The tool that reports and the tool that acts disagreed about the population, and
the acting one was blind to the only feature worth acting on. Both now compute
it over candidates only, through one shared function
(`features.within_query_variance`), so they cannot drift again.

This is the same class of bug as everything else in section 2, committed while
writing the section that catalogues the class.

### The promotion was refused

44 features, `has_hint` gone, train-time `pairwise_rank` capture **46.5% ->
67.6%** on seed 42. `app.retrain` rejected it:

```
challenger's worst pick is 3.31x the best available plan against the
champion's 1.41x -- a better mean does not buy a worse worst case
```

Which is 2.63's verdict, on an unrelated challenger, for the second time. Two
independent changes have now improved the mean and worsened the tail, and the
promotion rule has declined both.

That is a pattern rather than a coincidence, and it is the most useful thing in
this section. Every improvement this project has found lately makes the model
better on average by making it more willing to deviate -- and the tail is
exactly where deviating costs. The gate reads a point prediction (2.63), the
conformal bound covers 75% rather than 90% (2.66), and the ranker has none of
the ensemble's diagnostics. Until one of those changes, "better mean, worse
worst case" is the only shape an improvement can have here.

## 2.71 Two selectors, both mediocre, better together

2.70 ended on a pattern: every recent change improves the mean and worsens the
tail, and the promotion rule has declined two of them for it. That is a
statement about *what kind* of fix is needed, and it points at the item P2 had
been carrying for weeks -- "give the ranker the diagnostics the ensemble has".

### Most of that list cannot be given

The diagnostics named were `uncertainty_scale`, the conformal bound, the
calibration sweep and the promotion score. Every one of them is a **per-bundle
scalar**: one number for the whole model.

2.69's rule disposes of them before anything is built. A feature with one value
per model has one value per query, and the ranker is a within-query comparator,
so none of them can change a single decision. Not unhelpful -- *unusable*.

Two things vary per candidate. The first is the ranker's own margin over the
runner-up, which is its statement of how close the call was and which no part
of the gate reads. Measured over 160 held-out decisions:

| margin bucket | n | median x native | regressed | worst |
|---|---|---|---|---|
| thin | 53 | 0.834x | 19% | 1.23x |
| middling | 54 | 0.813x | 19% | 1.27x |
| wide | 53 | 0.824x | 25% | **2.22x** |

Spearman(margin, ratio vs native) = **+0.021**. No signal, and what little there
is points the wrong way -- the worst regression and the highest regression rate
are both in the *widest*-margin bucket. So the confidence gate that suggests
itself does not get built, for the same reason 2.61's `pg_overexplain` features
did not.

### The second one is the ensemble itself

Within a query, across 96 held-out queries and 6 splits:

| | median Spearman | right sign on |
|---|---|---|
| ensemble mean vs latency | **+0.680** | 93/96 |
| ranker score vs latency | **-0.660** | 95/96 |
| ensemble spread vs its own error | +0.212 | 72/96 |

The first two are the finding. The ensemble and the ranker train on the same
rows and the same features, so the intuition is that one must dominate. They do
not: they are of a strength, and they get there differently. 2.43 already said
why -- the ensemble predicts a *level* and the ranker predicts an *order*, and
uniform error cancels in a comparison and not in a prediction.

Two comparably good, differently-wrong rankings is the textbook case for
aggregating rather than choosing.

### Borda

Rank the candidates by predicted ratio, rank them by Copeland score, add the two
ranks, take the minimum. No weight to tune -- which matters, because 2.66
measured a single split worth up to 64.9 points and there is no held-out budget
here to fit a blending constant with. A weighted sum of a log-ratio and a
win-share would also be adding two things that are not in the same units.

160 held-out decisions over 10 query-level splits:

| selector | served | captured | worst regression |
|---|---|---|---|
| native | 23,526 ms | 0.0% | 1.00x |
| `pairwise_rank` (shipping) | 19,455 ms | 52.1% | **2.22x** |
| `greedy` | 19,250 ms | 54.7% | 1.64x |
| **`borda`** | **18,608 ms** | **63.0%** | **1.64x** |
| oracle | 15,715 ms | 100.0% | — |

**+10.9 points over the shipping policy, and the worst case improves at the same
time.** That is the first change in this stretch of the project to move both,
and 2.70 predicted the shape of what would: not a better single model, but a
second opinion on the one decision that matters.

### The arm that looked better and was not

The obvious version is stricter: serve the ranker's pick only if the ensemble
also ranks it in the top three, otherwise keep native. It does bound the risk --
worst case 1.64x, and it deviates on only **53%** of queries instead of 100% --
and it captures **42.4%**, eleven points *worse* than the policy it was meant to
improve.

Declining is not free. Every query it refuses is headroom left on the table, and
on this workload the refusals cost more than the regressions they avoid. Borda
gets the same worst case without refusing anything, because it is choosing a
better candidate rather than choosing less often.

### Ties are not preferences

The first implementation ranked by `sorted()` position, so two candidates a
selector scored identically got ranks 0 and 1. That invents a preference neither
selector expressed, and worse, it silently disables the cost tie-break: two
candidates both selectors rate equally come out with sums 0 and 2 rather than
tying, so PostgreSQL's estimate never gets consulted. Averaging ranks over ties
took `borda` from 61.5% to **63.0%**.

### Wiring, which is where this normally goes wrong

`pairwise_rank` was spelled out as a literal in seven files -- the autotune
sweep, the benchmark's `--policy`, the calibration replay, the promotion
yardstick, the settings API, the experiment runner and the training evaluation.
Adding a policy to six of them is exactly 2.50: a `gate_mode` the sweep offered
and the API ignored, and 2.43: a gate calibrated under a policy the server does
not run.

They all read `learned.RANKER_POLICIES` now, and `tests/test_borda.py` asserts
that each of the six scores `borda` *as borda* rather than as the plain ranker.

### And the sweep does not recommend it

That wiring is what makes the next paragraph possible, so it earned itself
immediately. With `borda` in the grid, `app.autotune` swept 40 configurations
and put it **second**:

| policy | extra families | net | deviates | regresses | worst |
|---|---|---|---|---|---|
| `pairwise_rank` | `resource` | **+4.3%** | 81% | 15% | 1.19x |
| `pairwise_rank` | none | +4.2% | 69% | 18% | 1.19x |
| `borda` | `resource` | +2.9% | **44%** | **14%** | **1.00x** |
| `greedy` | none | +0.0% | 6% | 0% | 1.00x |

The two measurements are not in conflict; they answer different questions. The
table above is **gated**, and the 63.0% figure is what the selector does when
allowed to act on every query. Borda deviates on 44% where the ranker deviates
on 81%, and every query it declines is headroom it does not collect.

Which is the same trade the `consensus` arm lost on, arriving from the other
direction -- and it says the binding constraint has moved. Borda is the better
*selector* and the gate will not authorise it, because the gate reads the
ensemble's point prediction for whichever candidate is put forward, and borda's
picks are compromises: rated well by both selectors, rated extreme by neither.
A gate thresholded on predicted gain systematically under-authorises exactly the
candidates that aggregation was introduced to find.

So `borda` ships as an option, off, with its number written down. Promoting it
needs the gate to read something other than a point estimate, which is the item
2.63 raised and nothing has yet done.

### A tenth of a point is not a reason to widen the action space

The same run recommended exporting `INCLUDE_TUNING=1`, on **4.3% against 4.2%**.

`MIN_POLICY_SWITCH_POINTS` already refuses that for the policy axis -- it exists
because this tool once recommended a policy switch on 0.8 points, two lines after
printing that a difference that size was inside its own spread. The action space
had no such guard, and it has the same property: turning a family on widens what
collection must execute and what the model must score, and 2.63 measured a wider
space raising the ceiling and the risk together. 2.67 measured this exact family
at 0.0 ms.

Same floor now applies to both, read from `planner`'s own flags so "the
incumbent" means what runs rather than what a constant says runs.

## 2.72 Eighty-five percent of a random workload has nothing to win

Every capture figure this project reports is measured on a workload that was
screened. `app.workload_generator` produced 45 candidates and 27 survived
vetting, on three rules:

    native under 8 ms       affordability -- a percentage of nothing
    native over 900 ms      affordability -- collection becomes unaffordable
    headroom under 8%       selection on the outcome

The first two bound what is practical to measure. The third selects queries by
the answer, and a capture ratio computed over its survivors is conditioned on
there having been something to capture.

`app.benchmark` has reported the effect for a while -- the gap between the
screened group and the hand-written one, most recently **+14.8 points** -- but
both of those groups were written by a person, so both carry somebody's idea of
an interesting query and only one of them admits it. `app.selection_effect`
removes the third rule instead: draw generated queries, keep everything
affordable, and look at the distribution.

Forty queries, all of them affordable:

| group | n | mean headroom | median |
|---|---|---|---|
| everything | 40 | **3.2%** | **0.0%** |
| kept by the >=8% filter | **6** | 16.0% | 13.8% |
| rejected by it | 34 | 0.9% | 0.0% |

**Six of forty.** The median generated query has *zero* headroom -- native is
already the fastest plan the whole action space reaches. By latency rather than
by query, 3.6% of all the generated latency is reachable against 14.8% of the
filtered subset's.

**Selection effect on generated queries: +12.8 points**, and it is not a
correction to apply to the headline. It is the headline's population.

### This is the same finding as the extended statistics

2.57 measured three `CREATE STATISTICS` statements removing 98% of the
opportunity on the queries this schema was built around. This measures 85% of a
random draw having no opportunity to begin with. Different mechanisms, one
conclusion: **the queries where a learned plan-picker earns anything are rare,
and finding them is most of the work.**

That is not a reason not to build one. A 40% improvement on 15% of a workload is
a real 6%, and the queries in question are usually the slow ones -- the 3.2%
mean is dragged down by a long tail of queries that were already fast. It is a
reason to be exact about which number is being quoted, which is why
`app.benchmark` leads with the figures nobody selected for.

### So `data/extended_stats.sql` ships, and not in `schema.sql`

The decision this leaves is whether the bundled schema should create the
extended statistics. Both answers are defensible and they answer different
questions:

- **On a real database, run them.** Three statements beat this entire project on
  cost, coverage and risk: they fix every query against those columns forever,
  including ones the optimizer never sees, and they cannot regress a query the
  way a wrong hint can. Any honest recommendation puts this first, and
  `app.advisor` already generates them for an arbitrary schema.
- **In the bundled demo, do not.** A layer that corrects the optimizer cannot be
  evaluated on a workload the optimizer already gets right. With statistics on,
  headroom is 0.4% and the benchmark measures noise.

So they ship as `data/extended_stats.sql` -- one command, reversible -- with the
trade-off written at the top of the file and in the README rather than left for
someone to discover. The dishonest version of this project is the same code with
that file deleted and the paragraph unwritten.

## 2.73 A gate belongs to a model, not to a path

`models/gate.json` has recorded its provenance since 2.50 -- the policy it was
swept under, the action space, the moment it was applied -- and
`app.diagnostics` compares each against what the server actually does. Those
checks were built after 2.43, where a gate calibrated under `greedy` was served
alongside a `pairwise_rank` selector and the two differed in *sign* on held-out
queries.

After 2.69's promotion, diagnostics reported:

```
  swept under policy  : pairwise_rank
  server selects with : pairwise_rank
  swept over families : cardinality, join_order, operator_toggle
  server offers       : cardinality, join_order, operator_toggle
=== findings ===
  [medium] uncertainty_understated
  [medium] error_comparable_to_headroom
```

Green on both drift checks. Two lines above, in the same output:

```
  calibrated on       : 15 queries
```

The deployed model held out **16**. The gate belonged to its predecessor.

### Why every check passed

`calibrated_against_bundle` records **`models/plan_selector.pkl`**. That is a
path, and `model_store.promote()` writes a new model to the same path. The field
was true before the promotion and true after, and it stopped describing the same
object somewhere in between.

The policy check passed because the policy did not change. The families check
passed because the action space did not change. Nothing compared the *model*,
because the only thing recorded about it could not change.

This is 2.65's shape again -- a defensible field doing less than its name says --
with the twist that the field was added specifically to catch this class of
problem and was defeated by an aliasing question nobody asked.

### The fix

`calibrate.bundle_identity` digests the feature columns, the sorted held-out
list and the target: the things a threshold's validity actually depends on.
Twelve hex characters, stamped into the gate on `--apply`, compared by
`app.diagnostics` against the bundle it is reporting on.

Not the pickle bytes, which are not stable across library versions, and not a
timestamp, which would flag a re-calibration of an unchanged model as drift.

A gate file written before the field existed reports **low: this cannot be
checked** rather than passing silently. Absent is not the same as matching, and
reporting green on an unanswerable question is the failure this section is
about.

### And a second one fell out of it

Re-running `python -m app.calibrate --apply` to fix the drift produced:

```
  swept over families : None
```

`app.autotune --apply` records `calibrated_with_families` and
`app.calibrate --apply` did not. Two entry points write the same file, one of
them stamps a field the other omits, and using the simpler tool erased a drift
check that had been working -- silently, because an absent field read as "no
information" rather than as a regression.

Both routes stamp both fields now, and `tests/test_gate_provenance.py` counts
the occurrences rather than trusting the next person to remember. The served
gate has been re-calibrated against the model actually deployed:

```
calibrated_on_n_queries    : 16
calibrated_with_policy     : pairwise_rank
calibrated_with_families   : cardinality, join_order, operator_toggle
calibrated_against_model   : 78ae980a7ab9
```

## 2.74 A flag with a paragraph of documentation and no effect

Thirteen `INCLUDE_*` environment variables widen the action space. Each is two
mentions in `planner.py` -- a definition and one read -- which is what a working
flag and a dead one both look like from the outside.

So: switch all thirteen off, switch each on alone, count what it generates.

```
all flags off: 20 hints

flag                          new hints                 families they add
INCLUDE_BUSHY                         1                      join_order:1
INCLUDE_COST_MODEL                    7                        resource:7
INCLUDE_DISABLE_INDEX                 0                                 —
INCLUDE_ENABLE_SETTINGS               3                 operator_toggle:3
...
INCLUDE_TUNING                        6                        resource:6
```

`DisableIndex(alias index)` asks the planner to search as though an index had
been dropped. It arrived with the PostgreSQL 18 upgrade, has a
twenty-line docstring recording that the two-argument form is required and how
each version fails differently, and a measurement showing a query going from an
8.4-cost index scan to a 4,059-cost sequential scan under it.

It generated nothing. Not an error, not a warning -- zero candidates.

### Why

Both index families need the alias-to-index catalog map, and one function builds
it:

```python
if not INCLUDE_INDEX_PINNING or cur is None:
    return None
```

`generate_candidates` then calls the DisableIndex generator only
`if include_disable_index and table_indexes`. With pinning off the map was
`None`, so the second family was silently switched off by the first flag's
absence. The docstring even explains the guard -- "missing either the flag or
the cursor means the family generates nothing" -- and that sentence was written
about index *pinning*, while the code it describes governed two families.

This is 2.53's shape: a setting with documentation and no effect. And it is
worse than a dead flag, because `app.autotune` prices a family by switching it
on and replaying (2.67). A family that generates nothing when switched on is one
the sweep is *obliged* to report as worthless, with a number attached.

### The fix has an inverted twin, and it nearly shipped

Building the map for either flag is one line. It would also have introduced the
mirror-image bug, because the pinning generator was gated on nothing at all:

```python
if table_indexes:
    candidates += generate_index_candidates(table_indexes)
```

That was correct exactly while the map's existence *was* the pinning flag. Make
the map available to a second family and asking for `DisableIndex` silently
delivers index pinning too -- one flag, two families, the same class of error
pointing the other way. `include_index_pinning` is now an explicit parameter,
defaulting to `True` so that callers passing a map and no flag keep meaning
"pin".

### Two more fell out of it

**`tests/test_family_guard.py` failed, and was right.** It compares the family
switches serving forwards against the ones collection forwards, because a model
trained on a family the server does not offer is 2.9's train/serve skew. The new
parameter defaults to True, so `app.collect_data` -- which passes the map --
would have collected a family serving gates off. A guard written for one
specific mistake caught its recurrence in a place nobody was looking.

**`app.audit_binding`'s "FULL action space" was not full.** It answers "did every
hint the optimizer generates actually bind?", which exists because
pg_hint_plan reports a rejected directive at `INFO` severity and the default
`client_min_messages` discards it (2.20). Its `FULL` dict listed eleven family
switches and not `include_disable_index` -- so the family whose keyword did not
exist before pg_hint_plan 1.8.1, the single most likely one to be silently
rejected, was never audited. `family_of` did not recognise `DisableIndex(`
either, and would have labelled it a join order.

Both fixed: 5,754 candidates audited before, **5,884** now, and all of them
bind.

### The pattern

Three tools in this project prove a family works: the sweep prices it, the audit
proves its hints bind, and the guard proves collection and serving agree about
it. All three were running. All three reported success on a family that could
not be switched on, because each was answering a narrower question than its name
implies -- and the flag test that found it took twenty lines and had never been
written.

## 2.75 The sweep could not see two of the eight families

`app.autotune`'s action-space axis is built from two lists:

```python
ALWAYS_ON         = {"join_order", "operator_toggle", "cardinality"}
OPTIONAL_FAMILIES = ("resource", "join_method", "scan_method")
```

`hint_family` names **eight** families. `memoize` and `parallel` are in neither
list, and `calibrate._servable(candidates, allowed=space)` drops every candidate
whose family is not in the set it is handed.

So with `INCLUDE_MEMOIZE=1`, the server offers the family, **every arm of the
sweep silently excludes it**, and the tool recommends a gate calibrated against
a narrower action space than the one being served.

That is 2.47's finding, in the tool written after 2.47 to prevent it. The two
lists that had to agree -- `hint_family`'s vocabulary and this tuple -- were
four hundred lines apart in different files, and nothing compared them.

### Completing the axis is not free, so it is derived instead

Five optional families is 32 subsets against 8, and 32 spaces x 5 policies x the
threshold grid is four times the work -- much of it comparing an arm against an
identical copy of itself, because a family with no rows in the log scores
exactly like the space without it.

`available_families()` reads the families actually present in the rows being
scored and sweeps only those. On the current log:

```
all optional families : ('resource', 'join_method', 'scan_method', 'memoize', 'parallel')
present in the log    : ('resource', 'scan_method')
skipped (no rows)     : ('join_method', 'memoize', 'parallel')
spaces the axis tries : 4   (was 8)
```

Complete **and** cheaper than the incomplete version it replaces.

### Which corrects 2.67

That section reported, from the same sweep:

| policy | extra families | net |
|---|---|---|
| `pairwise_rank` | none | +5.27% |
| `pairwise_rank` | `join_method` | +5.27% |

and read the identical figures as the family earning nothing. It earned nothing
because **there were no join-method rows in the log**. The sweep was comparing
an arm against a copy of itself and reporting the tie as a measurement.

The report now says so:

```
not on the axis -- no rows in the log to score them: join_method, memoize,
parallel. Collect with the family on before reading anything into its absence;
a family with no rows scores identically to the space without it, which is not
a measurement of the family.
```

"Not measured" and "measured worthless" are different answers. 2.67's verdict on
`resource` and `scan_method` stands -- those families are in the log. Its
verdict on `join_method` was never a verdict.

### Two more, one of them mine

**`families_env` covered three families.** It turns the winning action space into
the environment variables to export, and it hardcoded the same three. A sweep
that can recommend `memoize` and then prints instructions that leave it off is
worse than one that never sweeps it -- which is what
`tests/test_autotune.py::test_the_families_map_back_to_the_environment_that_produces_them`
had been written to forbid, and it failed the moment the axis grew. A test
written for one specific mistake, catching its recurrence two years of commits
later.

**And the new function repeated 2.70's mistake within the hour.** It began as

```python
from app.calibrate import _grouped_rows
```

which copies the function object into this module's namespace. Every test in
`test_autotune.py` supplies its rows by replacing `calibrate._grouped_rows`, and
so does `sweep`. The copied binding ignored both -- so the axis would have been
derived from the real database while being scored against the stub, and the test
suite went from 14 seconds to 99 while quietly measuring the wrong thing.

The symptom was runtime. The bug was the same question 2.70 raised and this file
had already written down: **which population is this computed over?** Calling it
through the module fixes it, and matching `sweep`'s keyword-argument convention
was needed too -- same function, same argument, different calling convention is
enough to miss a substitution.

## 2.76 The same rows are good training data and a false ceiling

`optimizer/prefilter.py` refuses to plan candidates the baseline plan *proves*
cannot change it -- `Set(enable_mergejoin off)` on a plan with no Merge Join
node. `planner.candidate_hints` runs it. `app.collect_data` calls
`generate_candidates` directly and does not.

So collection has, for the project's whole history, executed and logged
candidates serving never offers: **188 of 1,304** servable candidates on the
synthetic log, **14.4%**.

### The first reading was wrong

The obvious way to find this is to ask whether the prefilter ever drops a
*winner*, and the answer looked alarming: on **10 of 62** queries with a real
win, the winning hint was one the prefilter would have refused to plan --
`ctl_3w_recent` at 38.0% headroom, `gen_17_3w` at 34.2%.

Read one way that says the prefilter is too aggressive and has been costing
measured headroom. So: execute it. Plan each of the ten with and without the
hint and compare `plan_fingerprint`.

```
query                         plan changes?  native ms  hinted ms   ratio
gen_16_3w                                no     3561.8      787.1   0.22x
ctl_supplier_rating                      no       26.9       20.3   0.76x
corr_brand_band                          no      137.5      143.5   1.04x
...
the plan changed on 0/10 of them.
```

**Zero.** The prefilter's premise is exactly right -- `enable_X = off` only
penalises plans containing X, so a plan without X stays cheapest and comes back
byte-identical. What the log calls a 78% win on `gen_16_3w` is 3,561 ms and
787 ms for **the same plan**, which is a cache effect with a hint attached.

The finding inverts: the prefilter is sound, and the log's oracle is
contaminated.

### The oracle is a minimum over noisy draws

`min()` is biased low even when every individual timing is unbiased, so a
candidate that *is* native competes to be the ceiling native is measured
against -- and wins on **11 of 65** queries, 16.9%.

| ceiling | total | headroom |
|---|---|---|
| native | 10,206 ms | — |
| oracle as logged | 6,815 ms | **33.2%** |
| oracle over real candidates | 6,852 ms | **32.9%** |

The aggregate is 0.4 points. The per-query effect is 78%.

### And then the fix was wrong too

Dropping the rows is one line, and it is the obvious conclusion: they are
duplicate native measurements, serving never offers them, 2.9 says train on what
you serve.

Sixteen paired splits:

| | mean | better on | sign test |
|---|---|---|---|
| typical multiplicative error | **+0.015** | **1/16** | **p = 0.001** |
| `pairwise_rank` captured | +1.8 | 9/16 | p = 0.80 |
| `greedy` captured | -1.2 | 7/16 | p = 0.80 |

Dropping them makes the model **worse**, significantly, and does nothing legible
to headroom captured -- the per-split swings run from -52 to +63 points, which
is 2.66's range rather than a signal.

The reading that fits: a row saying "this hint changed nothing, and the result
was 1.0x" is real evidence about the **level**, and the level is what the safety
gate reads. It is 2.70 again -- removing the ranker's baseline pairs cost 4.2
points for the same reason. Both times the rows that looked like duplicates were
anchoring the scale.

### Where it actually belongs

The rows are legitimate training data and an illegitimate ceiling, so the filter
goes where the ceiling is computed and nowhere else:

- `app.collect_data` keeps collecting them, with the measurement written where
  the decision is, so the next reader does not "fix" it.
- `train.train(drop_provable_noops=False)` -- the option exists so the claim
  above is a paired measurement rather than a code diff.
- `train._evaluate_selection` drops them before computing the oracle, alongside
  the 2.47 filters it already runs.

Three defaults in this project are now set *against* a good argument by a
measurement that contradicted it: `rank_on_candidates_only`, the `has_hint`
ablation, and this. Each carries its numbers in a comment at the point of
decision, because the argument for changing them will look just as good next
time.

### And it was fixed in one place out of three

`_evaluate_selection` got the filter. Two other functions compute a *best
available* over the same pool:

| | quantity | contaminated? |
|---|---|---|
| `train._evaluate_selection` | the oracle every capture ratio divides by | yes |
| `retrain._score_bundle` | `worst_ratio_vs_best`, which rejected two challengers today | yes |
| `calibrate.evaluate_setting` | the gate's net improvement per deviation | yes |

Measured on the promotion yardstick specifically: **11 of 65** queries have a
phantom ceiling, and the champion's worst ratio moves **1.48x to 1.45x**. Small
enough not to have changed today's verdict, and it decided a promotion.

That is 2.47 exactly -- an action-space filter present in one scorer and absent
from the others -- found then in three places and fixed three times. So this one
lives in a single function, `calibrate.without_provable_noops`, and
`tests/test_oracle_noops.py` asserts each of the three calls it and that nobody
grows a second copy.

## 2.77 A block that can never lift

`planner.plan_query` carries this, above the query-level block:

> A blocked query never runs the learned path again, so it accumulates no
> evidence about it. The block then lifts purely because the old evidence ages
> out of `evidence_since()`, and the query is retried with whatever model is
> deployed -- having learned nothing in between. **That is a timer, not a
> decision.**

Shadow execution is the fix: a sampled fraction of blocked requests plans the
learned path anyway, executes it for measurement, and never serves it. The
argument is exactly right and it was applied to one of the two kinds of block.

### Family blocks are a ratchet

The guard blocks whole *queries* and it blocks hint *families* on a query
(`allowed_hints`). For families, the same reasoning produces a closed loop:

1. a family regresses on a query, so it is blocked;
2. `plan_query` calls `guard.allowed_hints` and the family is removed from the
   action space -- **including on the shadow path**, which called it
   unconditionally;
3. so the family is never planned, by any path, so it produces no row;
4. so nothing can ever clear it.

The block lifts only when the regressing rows age out of the evidence window. A
timer, not a decision -- twenty lines below the sentence saying so.

### And the evidence would have been invisible anyway

Two SQL strings decide what counts as evidence:

```sql
REGRESSION_SQL         WHERE (is_chosen OR selector_used = 'shadow')
FAMILY_REGRESSION_SQL  WHERE is_chosen
```

So even a shadow row carrying a blocked family -- which step 2 made impossible,
but had it not -- would not have been counted by the thing that blocks families.

Two independent halves of the same mistake, and **either one alone leaves the
ratchet intact**. That is why it survived: fixing the visible half would have
changed nothing observable and looked like the wrong diagnosis.

### Latent, and that is why it is here

The guard currently holds **zero** family blocks and the log **zero** shadow
rows inside the evidence window. Nothing has been affected.

Which is the point. A block that never lifts and a block that lifts slowly are
indistinguishable until somebody waits, and the evidence window is measured in
days. The failure mode is a family quietly excluded from a query's action space
forever, on evidence from one bad afternoon -- and every measurement of that
query afterwards reporting a smaller ceiling, correctly, for a reason nobody
would think to look for.

### The fix, both halves

`plan_query` skips `allowed_hints` when `bypass_block` is set, so the shadow
path plans blocked families; `FAMILY_REGRESSION_SQL` counts shadow rows, so it
can see them. `bypass_block` is passed by exactly one caller, asserted by a
test, because a fix that also released the brake would be worse than the
ratchet.

Moving `SHADOW_SELECTOR` above the first query that interpolates it was needed
too -- it had been declared beside the query that used it, which was fine while
only one did.

## 2.78 Two definitions of "which query is this", one comment saying they agree

`logging_store.query_fingerprint` carried this:

> Must stay in sync with `QUERY_KEY_SQL` below, which reproduces this same id in
> SQL for rows written before ids were assigned.

The two rules:

```
python:  query_template(sql) if GROUP_BY_TEMPLATE else whitespace(sql)
sql:     regexp_replace(btrim(sql_text), '\s+', ' ', 'g')
```

The SQL reproduces the `GROUP_BY_TEMPLATE = False` branch. The setting defaults
to **True**, and on that branch `query_template` also strips literals and
`/*+ ... */` hints. So on the default configuration:

| | python | sql |
|---|---|---|
| same template, different literal | same | **different** |
| a query against its hinted form | same | **different** |
| whitespace only | same | same |

Two of three.

### Latent, and it stays that way deliberately

`COALESCE` prefers the recorded `query_id`, so the fallback only decides how
rows *without* one are grouped -- and there are **0 of 33,634**. Nothing is
affected today.

The obvious fix is to implement `query_template` in SQL. That was considered and
rejected, and the reason is the rest of this section of the writeup: 2.75 was
two family lists that had to agree and stopped, 2.76 was one filter in three
scorers, 2.77 was two SQL predicates deciding what counts as evidence. Adding a
second implementation of *query identity* -- a pair of Python regexes,
transliterated -- to close a path nothing currently takes would be buying the
exact failure mode this night was spent removing.

So the comment is corrected instead. It now says which branch the fallback
implements, gives the three cases and their answers, records that it is
unreachable today and what would make it reachable, and points at
`tests/test_query_key_fallback.py`, which pins both the agreements and the
disagreements.

### The part worth generalising

A wrong comment is worse than no comment, and specifically worse in a way that
compounds: **it is the reason nobody checks.** "Must stay in sync with X" reads
as an invariant somebody is maintaining. Four sections of this writeup were
found by asking whether two things that must agree still do, and this one had a
note asserting the answer.

The rule that would have caught it is the one 2.70 arrived at from the other
direction -- *which population is this computed over?* -- applied to a claim
rather than to code.

## 2.79 The recommended threshold switched off the safety mechanism

2.63 left the gate's standing weakness open: it reads a point prediction, so a
confident wrong one is exactly the input it cannot decline. Three replacements
have now been measured.

| input | signal | verdict |
|---|---|---|
| the ranker's margin over the runner-up | Spearman **+0.021**, worst case in the *widest* bucket | not built (2.71) |
| the ensemble's spread | +0.212, and it understates the error 3.2x | not built |
| **the two selectors disagreeing** | see below | built, defaulted off |

### Disagreement predicts variance, which is what a gate is for

2.71 found the ensemble and the ranker comparably strong within a query (+0.680
and -0.660) and wrong in different places, which is why aggregating them beat
either. So where they *agree*, both say the same thing; where they disagree, at
least one is wrong and the gate cannot tell which.

Bucketed by the ensemble's rank of the ranker's pick, over **384** held-out
decisions across 24 splits:

| | n | median x native | regressed | worst |
|---|---|---|---|---|
| agree | 129 | 0.813x | **13%** | **1.64x** |
| middling | 126 | 0.858x | 21% | 3.22x |
| disagree | 129 | **0.753x** | **22%** | 2.22x |

Spearman against the served ratio is **-0.033** -- nothing. It does not predict
whether the plan will be faster. It predicts *how sure you can be*: the disagree
bucket has the **best** median and twice the regression rate. Bounding that,
rather than maximising the mean, is the gate's entire job.

Spent as `caution` rather than as a veto, because the veto form was measured in
2.71: `consensus` bounded the worst case identically and captured **11 points
less**, since every query it refuses is headroom left on the table.

### And it does nothing, for a reason worth more than the feature

Swept through the replay:

```
DISAGREEMENT_CAUTION=0.0   z=2.0 gain=0.00   69% deviate, 18% regress, +4.2%
DISAGREEMENT_CAUTION=0.5   z=2.0 gain=0.00   69% deviate, 18% regress, +4.2%
DISAGREEMENT_CAUTION=1.0   z=2.0 gain=0.05   62% deviate, 20% regress, +3.5%
```

Byte-identical at 0.5, worse at 1.0. The reason:

```
required = 1.0 - min_relative_gain * caution
```

The deployed gate has **`min_relative_gain = 0`**. At zero that expression is
1.0 for *every* caution — so the whole caution mechanism is inert on the
condition that decides whether a win is worth taking. Not just this new term:
the regression guard's **2.0** for a query it has no history for, documented at
length, with 2.18's 96% veto rate as the story of getting it wrong. Only the
downside ceiling still responds to caution.

`calibrate.py`'s own docstring makes exactly this argument about the other knob
— "a gate with `z=0` also multiplies the model's uncertainty by zero, so the
safety mechanism was, in effect, switched off" — and nobody made it for
`min_relative_gain`. `_dominates` prefers a higher gain among *equals*, and zero
won outright on net improvement, so the tie-break never engaged.

**It was deployed earlier the same night**, by a `calibrate --apply` run in this
session fixing 2.73's stale gate. The previous gate had `min_relative_gain =
0.02`.

### What shipped

The sweep now says so when it recommends zero, in the recommendation itself
rather than in a docstring. The disagreement term ships as a mechanism with its
measurement attached and its default at **0**, because a term that multiplies an
inert quantity cannot be evaluated — turning it on is a decision to make after
`min_relative_gain` is non-zero, not before. `calibrate.evaluate_setting`
computes the same quantity the served gate does, so the sweep is not recommending
thresholds for a system nobody runs (2.43), and a test pins the two together.

The general shape: a sweep optimises what it measures, and "this setting makes a
different safety mechanism inert" is not in its objective. Three of the four
knobs interact and the grid treats them as independent.

## 2.80 99.2% was a statement about the action space

`docs/JOB_RESULTS.md` has led with **99.2% of available headroom captured** on
the Join Order Benchmark since the narrow-space collection. 2.56 widened the
action space, found the ceiling moving under it on the queries collected so far,
and left the headline flagged as stale. Re-collecting all 30 JOB queries under
the wide space took **8h 22m** and 3,100 rows.

Same model class, same 8 held-out queries, same split seed:

| | old (narrow) | new (wide) | change |
|---|---|---|---|
| native | 3,960 ms | 2,715 ms | −1,246 |
| oracle (ceiling) | 696 ms | **520 ms** | **−176** |
| served by `pairwise_rank` | 723 ms | **718 ms** | **−5** |
| served **minus** oracle | 27 ms | **198 ms** | **+171** |
| captured | **99.2%** | **91.0%** | −8.2 |

### The model did not get worse. It did not change at all.

Served latency moved **5 ms**, which is nothing. The same model picks the same
plans and delivers the same result. What moved is the ceiling: 176 ms of plan
quality that did not previously exist, of which the model captured **none**.

So 99.2% was never a measurement of how well the model chooses. It was a
measurement of how little there was to choose between: the best plan in the
narrow space was **27 ms** better than the one being served, and capturing 99.2%
of 27 ms is easy. Widen the space by 176 ms and the same behaviour reads 91.0%.

This is the failure mode the oracle column exists to prevent, arriving through
the oracle column. A capture ratio is a fraction of an opportunity, and reporting
it without the size of the opportunity lets a narrow action space masquerade as a
good model -- which is exactly what 2.55 said about `5b` ("a query where no
candidate beats PostgreSQL. That was true of the narrow action space") applied to
the headline.

### And native moved 31%

3,960 ms to 2,715 ms for the same eight queries. Nothing about the queries or the
database changed; this is 2.32's between-run drift, on a real benchmark, at a
scale that dwarfs every effect this project argues about. It is why the table
above compares *within* each run and why the served-minus-oracle row is the one
to read: both numbers in it were measured in the same run, against each other.

Any cross-run comparison of JOB latencies in this project is worth about as much
as the 31%.

### `borda` takes some of it

Of the 198 ms now on the table, `borda` recovers **30 ms** -- 92.3% captured
against `pairwise_rank`'s 91.0%, the best of the five policies, and the same
ordering 2.71 measured on the synthetic workload. `greedy` and `risk_averse`
land between them.

That is the second independent workload on which aggregating the two selectors
beats either, and it is still not deployed, for 2.79's reason.

### What the headline should say

**91.0%**, with the ceiling beside it, and the note that the previous 99.2% was
measured against a ceiling 27 ms above the served plan. `docs/JOB_RESULTS.md`,
`README.md` and 2.55 updated.

The feature audit on the new bundle reports **75 of 142** features constant
within a query -- more than half, consistent with the 72 of 139 the narrow-space
bundle showed, so widening the action space did not change how much of the vector
can separate candidates.

## 2.81 The deterministic session was pinned away from what the server does

`measurement.make_deterministic` pins two session settings so a measurement is
repeatable. It is called by `app.collect_data`, `app.benchmark` and
`app.selection_effect` — and **not** by `app.main`. Serving runs on the server's
defaults, and every latency serving logs becomes a training label.

So the two lists have to agree, and one entry did not:

| | pinned | server default |
|---|---|---|
| `max_parallel_workers_per_gather` | 2 | 2 |
| `geqo_seed` | **0.5** | **0** |

### The reason for 0.5 was wrong

The test asserting it carried the argument:

> It defaults to 0, which sounds fixed and is not — GEQO reseeds per planning
> cycle. On a large join that turns a paired A/B into a comparison of two draws
> from a random search.

If true, that is a good reason to pin a non-default value: repeatability would
be worth the train/serve gap. Planning the same query **twelve** times at
`geqo_seed = 0`:

```
geqo_seed = 0     distinct plans over 12 plannings: 1
geqo_seed = 0.5   distinct plans over 12 plannings: 1
```

Zero is a fixed seed like any other, on PostgreSQL 18.6.

### But the setting itself is load-bearing

Worth establishing before changing it, since "probably inert" is how several
sections of this file begin. With GEQO forced on (`geqo_threshold = 2`,
`geqo_effort = 1`), four seeds give **four different join orders** for one query
— `movie_companies`, `cast_info`, `name` and `title` land in a different order
each time, each seed internally stable.

So: pinning is right, and among fixed values the server's own is free.
`geqo_seed` is now **0**, and both entries match what serving does.

### And a fact this project has been getting wrong

The measurement above needed a query where GEQO actually runs, which turned out
to be harder than expected. At the default `geqo_threshold = 12`, JOB's `17c`,
`17d` and `17e` plan identically under every seed — because **they are not
17-table queries**. `17` is the query family's *name*. `17e` joins **seven**
tables.

JOB's 17-table queries are the `29*` family. Counted across all 113: the largest
is 17 tables, the smallest 4, and **20 of 113** reach the GEQO threshold.

This project has only ever collected the **smallest 30**, none of which do. So
GEQO has never run on a single query in any measurement here, on either
database. `geqo_seed` is load-bearing in principle and has never been exercised
— which is exactly the class of latent difference the pin exists for, and the
reason the right value is the server's rather than an arbitrary one.

`PENDING.md`'s "the benchmark goes to 17-table joins" is true of JOB and was
being read as true of the queries collected. Corrected there.

## 2.82 A fresh install was slower than doing nothing

`PENDING.md` has listed cold start under "untested, not broken" for months:

> **Cold start.** Before a model exists, selection falls back to the cost
> heuristic — which JOB shows can be *worse than native*.

Tested. It is broken, and the obvious fix is worse than the bug.

### What the no-model branch did

```python
if self.model is None or baseline_plan is None:
    index = self.select(candidate_plans)        # cost argmin; native NOT in the pool
    self.last_decision["fell_back_to_baseline"] = False
    if baseline_plan is not None and self._is_unsafe(chosen, baseline_plan):
        return baseline_plan
    return chosen
```

Native is not among the candidates being ranked, so there is no "is this better
than native?" test at all. The only thing in the way is `_is_unsafe`, and 2.29
already established that its cost test **cannot fire on PostgreSQL 18** — the
server stopped inflating disabled plans to `disable_cost` and says
`Disabled: true` instead. Its own docstring says so.

Replayed over the 65-query synthetic log:

| | served | vs native | deviates | of those, regress | worst |
|---|---|---|---|---|---|
| **as shipped** | 10,617 ms | **1.040x** | **65/65** | 23 (35%) | **5.32x** |
| never deviate | 10,206 ms | **1.000x** | 0/65 | — | 1.00x |

A fresh deployment deviated on **every query**, made the workload **4% slower**
than doing nothing, and ran one query at **5.3x**.

### The obvious repair makes it worse

Only deviate when PostgreSQL costs the candidate *below* native — use the one
signal available. Measured:

| | served | vs native | deviates | of those, regress |
|---|---|---|---|---|
| cheaper than native | 11,315 ms | **1.109x** | 12/65 | **10 (83%)** |
| cheaper by 10% | 11,342 ms | 1.111x | 10/65 | 9 (90%) |

Worse than the bug, and the reason is this project's own thesis turned on
itself. **The cost model's disagreement with reality is the signal the learned
model exists to exploit.** So "the estimate says this candidate is cheaper than
native" selects precisely for the queries where the estimate is wrong — and 83%
of the deviations it permits regress, against 35% for taking the argmin blindly.

A filter built from the quantity you have spent the whole project proving
unreliable will be anti-correlated with the truth, not merely uninformative.
That is worth stating because it looked like the safe, conservative change.

### What shipped

With no model, serve native. `select_plan`'s docstring already said so —
"defaulting to native unless there is *evidence* a hinted plan is better" — and
with no model there is no evidence. The branch now agrees with the contract it
sits under.

`_select_heuristic` stays: it is the Phase 0 reference point every evaluation
scores against, and `train._evaluate_selection` computes it directly from the
costs rather than through `select_plan`, so no baseline moves. The only thing
that changes is what a live query is served before the first model exists — and
`app.benchmark` still labels those rows `heuristic`, so they stay identifiable.

### Why it survived

Because it is the state a running system leaves in minutes. Collect, train,
promote, and nothing takes that branch again — the demo passes through cold
start on the way to the interesting part, and the interesting part is what gets
measured. It is only a user's first day that runs on it.

## 2.83 The benchmark was giving the model plans the server never offers

`app.diagnostics` has a section headed "is the gate keeping its promise?". On
this deployment it read:

```
regressions   promised 0.182   observed 0.190
deviations    promised 0.688   observed 0.323
```

The regression rate is accurate to a point. The deviation rate is out by a
factor of two, and **nothing flagged it** — the regression check fires when
observed exceeds promised by ten points, and the deviation check fired only when
the observed rate was *exactly zero*.

### Two hypotheses, both wrong

**The candidate budget.** `planner.budget_for` caps how many candidates a live
query is given and the replay scores every logged one, so the replay should have
more to choose from. Measured: 1,042 candidates in the replay against at most
965 servable, **93%**, and identical medians of 13. Not this.

**The population.** The gate was calibrated on 16 held-out queries and
diagnostics observes over everything. Measured: **28.8%** on held-out runs
against **31.0%** overall. Not this either.

### It was in the benchmark

The per-query log said it plainly:

```
corr_brand_category   predicted 0.55x native (pessimistically 0.66x, needs < 1.00x)
  -> safety veto: learned pick discarded, kept native plan
```

Both gate conditions satisfied — a predicted 45% speedup with a pessimistic
bound of 0.66x — and the query served native. Across the run, **43 queries were
vetoed and 36 of them looked like that**.

The veto firing is `_is_unsafe` catching a plan with **disabled nodes**:
one the planner was forbidden to build and built anyway, under a constraint it
could not satisfy. `planner.plan_query` drops those from the action space
outright (2.29). `app.benchmark` builds its own candidate list and never did.

They are not neutral candidates. On PG18 a disconnected four-table order costs
**20.5** against the correct plan's 6,990.8, so they look enormously cheap, the
model picks one, the veto discards it — and the query is served native, having
**lost its deviation entirely** rather than falling back to the next real
candidate.

Removing them: vetoes **43 → 30**. Thirteen queries got their deviation back.

### The fourth place, and the one that publishes

This is the same divergence as 2.47 (the calibration replay), and 2.76
(`retrain._score_bundle` and `train._evaluate_selection`). Three of those score
models. The fourth produces the numbers in `README.md`.

The pattern by now: *every* component that assembles a candidate pool has, at
some point, assembled a different one from the server's. The filters are
individually reasonable and were added one at a time to whichever consumer
someone was looking at.

### And the check that should have caught it

Diagnostics computed the discrepancy and printed it for months. The deviation
comparison is now two-sided: a gate deviating **below** its promise is leaving
headroom on the table, and it does so without producing a single regression to
notice — which is why the one-sided check was reasonable for regressions and
wrong here.

The finding also carries the diagnosis, so the next person to see it checks the
benchmark's action space before re-tuning a gate that was never the problem.

## 2.84 Two classifications of the same hint, on purpose

`PENDING.md` carried two items with the same shape for months:

> **A marker for bushy join orders.** `INCLUDE_BUSHY` was the one family 2.67
> could not price, because bushy hints are `Leading(((a b) (c d)))` and classify
> as `join_order`, so they are indistinguishable from linear ones in the log.

> **`DisableIndex` still cannot be priced separately.** 2.74 made the flag work,
> but its hints classify as `scan_method`, which the axis already sweeps.

A family the sweep cannot separate is one it cannot price. Turn the flag on and
every arm scores the same, which 2.75 established is not a measurement of the
family — it is the family's absence, reported as worthlessness.

### The fix is not a change to `hint_family`

The obvious move is to make `hint_family` return `bushy` and `disable_index`.
That would be a behaviour change to something else: `hint_family` has a second
consumer in `regression_guard.allowed_hints`, which blocks a whole *kind* of
action on a query that kind has already lost on.

If `Leading(a b c)` has regressed on a query, letting `Leading(((a b) (c d)))`
through afterwards on the grounds that it is a different family is not the
conservative reading. The coarse grouping is right there.

So there are now two classifications, differing by exactly one level of one
hierarchy:

| hint | `hint_family` (what the guard blocks by) | `action_family` (what the sweep prices) |
|---|---|---|
| `Leading(a b c)` | `join_order` | `join_order` |
| `Leading(((a b) (c d)))` | `join_order` | **`bushy`** |
| `SeqScan(o)` | `scan_method` | `scan_method` |
| `DisableIndex(o o_idx)` | `scan_method` | **`disable_index`** |
| everything else | same | same |

Two classifications is normally the bug this writeup keeps finding — 2.75, 2.77,
2.78 are all pairs of definitions that had to agree and stopped. The difference
is that these two are *supposed* to differ, so what has to be pinned is the
relationship rather than the equality: every action family maps to exactly one
hint family, and `tests/test_action_family.py` asserts it in both directions.

### Detected by structure, not by a marker

A bushy order is `Leading(` followed by another `(`. That is what pg_hint_plan
parses, so reading it back is reading the thing itself — a marker comment
attached at generation time would be one more pair that can drift, which is the
subject of three of the last five sections.

### What it unlocks

`_servable`, `served_families` and the sweep axis all read `action_family` now,
so `INCLUDE_BUSHY=1` no longer has its candidates silently folded into
`join_order` — the 2.75 bug, one level down, which was live and unnoticed for
exactly as long as the flag existed.

The sweep reports seven optional families and, on the current log, five as **not
scored**:

```
all optional  : resource, join_method, scan_method, memoize, parallel, bushy, disable_index
present in log: resource, scan_method
not scored    : join_method, memoize, parallel, bushy, disable_index
```

Which is the honest state: the tooling can now price all seven, and the data
exists for two. Collecting with the other five switched on is the remaining
step, and it is a cost in hours rather than a question.

## 2.85 Three quarters of the measured benefit was noise

`--protocol paired` (2.68) re-times native, served and oracle **alternately**
and takes medians, so between-run drift is common-mode and the oracle stops
being a minimum over noisy draws. It has now been run against the deploy
protocol on the same code, the same gate and the same hour — the only
difference is the timing.

| 65 queries | `deploy` (one pass) | `paired` (5 alternated) | change |
|---|---|---|---|
| native | 7,426 ms | 5,680 ms | −24% |
| served | 6,629 ms | 5,464 ms | −18% |
| oracle | 6,122 ms | 5,232 ms | −15% |
| **headroom** | **1,304 ms** (17.6% of native) | **448 ms** (7.9%) | **−66%** |
| **latency actually saved** | **797 ms** | **215 ms** | **−73%** |
| captured | 61.1% | **48.1%** |  |

Held-out queries, the number that says whether the model generalises:

| | `deploy` | `paired` |
|---|---|---|
| headroom | 213 ms | **63 ms** |
| saved | 85 ms | **12 ms** |
| captured | 40.0% | **18.4%** |

### What moved and why

**Native dropped 24%** between the two runs. Nothing about the queries or the
database changed; that is cache warmth, and it is the same effect 2.80 found at
31% on JOB. It cancels in a ratio, which is why the ratio is the thing to read.

**Headroom dropped 66%.** This does not cancel, because it is not a ratio of two
things measured the same way. `best_ms` is a **minimum** over candidate timings,
and a minimum over noisy draws is biased low even when every individual timing
is unbiased. Re-time the arms and the ceiling rises toward native. More than half
of the "available headroom" this project has reported was that bias.

**The saving dropped 73%.** 797 ms became 215 ms. This is the one that matters:
it is not a denominator effect, it is the measured benefit of running the system
shrinking by three quarters once each arm is timed five times instead of once.

2.68 predicted exactly this at single-query scale — a query reported regressing
at 1.273x that seven alternated repetitions put at 1.095x — and said "the
conclusion survived and its magnitude did not". At workload scale the conclusion
still survives: the system saves latency, on this workload, net. It saves about a
quarter of what a single pass says.

### The selection effect went to zero

`deploy` puts the curated group 7.8 points **below** the whole workload; `paired`
puts it 0.2 points **above**. The previously published figure was +14.8.

Three numbers for one quantity across three protocols is not a measurement of
anything, and the honest reading is that the selection effect is **not
resolvable at this sample size** — 27 filtered queries against 38 unfiltered,
with a per-query noise floor that 2.18 puts at 18.6%. 2.72's version of this
question is the one to quote, because it was measured over 40 queries with the
filter explicitly removed rather than inferred from two hand-written subsets.

### The caveat, stated plainly

One run each. `paired` prints `(single run -- spread unknown)` and means it.

What is not a matter of sampling: the direction. The `min` bias is one-sided by
construction, so re-timing can only lower the reported headroom, never raise it.
Both fixes in this section's neighbourhood push the same way — 2.83 removed
candidates the server never offers, this removes an optimistic ceiling — and
both make the project's numbers smaller.

That is the expected shape when measurement gets better, and it is worth saying
that neither correction was found by looking for a smaller number.

## 2.86 The transfer result was the narrow action space, not the model

2.64 is the strongest external-validity claim this project makes: a model
trained on the six-table synthetic schema, carried to JOB with **zero**
overlapping tables and every per-table feature slot reading zero, captured
**97.0%** of JOB's headroom against a locally-trained model's 98.4%.

2.80 flagged it as measured against the narrow ceiling and said "the ratio
between the two is the claim and it may well survive; the levels will not."

Re-run on the wide-action-space collection, with 2.76's filters applied:

| selector | total | of native | headroom captured |
|---|---|---|---|
| native (do nothing) | 104,246 ms | 100.0% | 0.0% |
| oracle | 24,768 ms | 23.8% | 100.0% |
| cost argmin | 100,655 ms | 96.6% | 4.5% |
| random pick | 131,093 ms | 125.8% | −33.8% |
| **carried synthetic model** | **108,435 ms** | **104.0%** | **−5.3%** |
| locally-trained JOB model | 27,534 ms | 26.4% | **96.5%** |

**97.0% became −5.3%.** The carried model now serves JOB 4% *slower* than doing
nothing. The ratio did not survive either: it is not 97.0 against 98.4, it is
−5.3 against 96.5.

### The local model is fine. The transfer is not.

96.5% for a model trained on this schema, so nothing about the wider action
space broke the approach. What broke is the claim that the *structural* half of
the feature vector is enough on its own.

### Why a narrow ceiling flattered it

Under the narrow action space almost every candidate was survivable — 2.80
measured the best plan there as **27 ms** better than the one being served
across eight held-out queries. A selector cannot do much damage choosing among
options that barely differ, so any selector scored well, and "97% of a 27 ms
opportunity" read as competence.

The wide space contains plans that are catastrophically bad as well as plans
that are much better. Ranking *that* needs to know something about the tables,
and the carried model knows nothing: every per-table slot is zero, by
construction, because the table names do not match.

### It is not picking at random, and that is the interesting part

Random selection lands at **125.8%** of native; the carried model at **104.0%**.
So the schema-agnostic features — costs, tree shape, join-method counts — carry
real signal and rank plans far better than chance. They are simply not enough to
clear the bar that matters, which is *native*, and 2.64's framing of "the
structural half travels" is right in direction and wrong about how far.

It also picks the exact best candidate on **4 of 30** queries, which is a real
skill and does not stop the aggregate being negative — the losses are larger
than the wins.

### What the claim should be

Not "works on any PostgreSQL database". A model carried to a schema it has never
seen ranks plans **well above chance and below native**, so deploying one is
worse than deploying nothing, and the useful version of the finding is the
warm-start question 2.64 never asked: does carrying a foreign model make local
*training* faster than starting cold? That is a different experiment and it has
not been run.

`README.md` and 2.64 updated. The two rows in 2.64's table are left in place with
a note, because the measurement was correct — it was the ceiling underneath it
that was not.

## 2.87 Under load, and the thing that made it safe was never written down

`PENDING.md` listed this under "untested, not broken":

> **Single-node, single-connection.** Every latency here is measured with no
> concurrency. Nothing is known about behaviour under load.

Two questions, and they are different kinds.

### Correctness: shared state across requests

`DecisionCache` takes a lock. `RegressionGuard` does not, and is read on every
request while `refresh()` replaces it on every settings change, promotion and
server start.

Hammered against a live database — six threads reading `is_blocked`,
`is_family_blocked`, `caution_multiplier` and `runtime_budget_ms` while a
seventh refreshed — **3,141,552 reads, zero errors.**

It is safe, and it is safe for a reason nothing recorded: `refresh` **rebinds**
each attribute to a fully-built dict rather than mutating one in place. A reader
sees the old dict or the new one, never a half-filled one.

Rewrite it as `self.blocked.clear()` followed by `update()` and it reads as an
obvious tidy-up, keeps every test passing, and introduces a genuine race — a
reader mid-refresh sees an empty block list and a query that should be blocked
is served its regressing plan. That is now a docstring and a test, because the
race would not fail reliably enough to be caught any other way.

**What is not protected**, stated rather than glossed: the four fields are
assigned in sequence, so a reader landing between two of them pairs a new
`blocked` with an old `observations`. The window is two bytecodes and the cost
is one decision using the previous refresh's caution. Bounded, and judged not
worth a snapshot object — but the same consistency argument the function already
makes about its own cutoff (`one boundary for every query below, resolved once`)
applies to readers across it, and was not made.

### Behaviour: does planning stop being cheap?

The economic case for the whole system is that choosing a plan costs less than
running one. Planning is many small round trips; execution is one large query.
Under load they could easily degrade differently.

`corr_4w_full_correlated`, both arms measured in the same run:

| concurrent | plan | execute | overhead | plan ×solo | exec ×solo |
|---|---|---|---|---|---|
| 1 | 50.1 ms | 114.1 ms | **44%** | 1.00x | 1.00x |
| 4 | 72.4 ms | 175.3 ms | **41%** | 1.44x | 1.54x |
| 8 | 130.3 ms | 277.1 ms | **47%** | 2.60x | 2.43x |

They degrade together — 2.60x against 2.43x at eight-way — and the overhead
ratio stays between **41% and 47%** throughout. The economic case does not get
worse under load, which is the answer that was wanted and not the one that was
expected.

The 44% is the more uncomfortable number, and it is not new: planning this query
costs nearly half what running it does, which is why `decision_cache.py` exists
and why `budget_for` scales the candidate count to what the query is worth. That
is a solo-path cost that repeat traffic pays once.

### What is still untested

Concurrency *of the serving path end to end* — this measured the planner and the
executor, not FastAPI, the connection pool, or the cache under contention. And
one machine, one PostgreSQL, no network. The claim this closes is narrow:
**shared optimizer state survives concurrent access, and planning overhead is
stable in proportion up to eight-way.**

## 2.88 A noise floor that costs nothing to measure

`PENDING.md` carried this for months, marked *free*:

> Buffer counts as a **per-comparison noise floor**. Two candidates with
> identical buffer counts did the same page work, so any latency difference
> between them is noise. Free — it is already in the log.

It is the third proposed use of buffers and the first that survives. 2.58
rejected them as a **label** — Spearman **0.341** against latency, so they
measure the wrong quantity. The train/serve skew rules them out as a
**feature**: they come from `EXPLAIN ANALYZE` and the serving path plans with
`analyze=False`, so the model would train on a column that is absent when it
matters. Neither objection touches an *offline, pairwise* use that decides what
a difference has to beat.

### It is computable, which was not obvious

`app.noise` measures the floor by re-executing one unchanged plan five times.
That is the definition of noise and it costs a full extra workload pass — and it
measures the **native** plan only, then applies the number to comparisons
between candidates.

Reading the log instead:

```
65 queries, 306 groups of plans doing identical page work, 1290 pairs.
Median disagreement 12.7%, worst 110.5%.
```

**Every one of the 65 queries** has at least one group, so this is not a
technique that works on a lucky subset.

### What it says that the global floor does not

The global floor is 2.18's **18.6%**. The per-query floors run from under 6% to
over 110%:

| query | floor | groups |
|---|---|---|
| `gen_27_5w` | **129.7%** | 6 |
| `gen_04_4w` | 118.2% | 5 |
| `gen_03_3w` | 109.5% | 3 |
| … | | |
| `shape_group_by_category` | 5.9% | 3 |

On `gen_27_5w`, two plans that read and wrote **the same pages** differed by more
than **2x**. A "2x speedup" measured on that query is indistinguishable from a
lucky moment — and the global 18.6% would have called it real six times over.
Meanwhile a 6% win on `shape_group_by_category` is dismissed by the global floor
and is three times that query's own.

The single number is wrong in both directions, and which direction depends on
the query.

### Why the floor is the widest group, not the average

It is a bar a difference has to clear. Averaging a query's groups lets its
noisiest pair through, which is the failure the floor exists to prevent. Stated
in the code and pinned by a test, because "take the mean" is the change someone
makes while tidying.

### What it does not do yet

`app.noise --from-buffers` reports it. Nothing consumes it: `stats.classify`
still grades against one `recommended_material_fraction` for the whole workload.

Wiring a per-query floor into the grading is a real change to what
`held_missed` and `deviated_win` mean, and it should be made against a re-run
benchmark rather than folded in beside eight other corrections. That is the
honest place to stop — the measurement exists, is free, and is written down.

## 2.89 Two parsers for one environment variable

Every action-space flag has two readers. `planner` reads it at import:

```python
INCLUDE_BUSHY = os.getenv("INCLUDE_BUSHY", "0") != "0"
```

and `settings.defaults()` reads the same variable through `_from_env`, whose
`_coerce` accepted `{"1", "true", "yes", "on"}`.

That is two rules, and they disagreed on five of ten plausible values:

| `INCLUDE_BUSHY=` | planner | settings |
|---|---|---|
| `1`, `true`, `yes`, `on` | True | True |
| `0` | False | False |
| **`false`** | **True** | False |
| **`no`** | **True** | False |
| **`off`** | **True** | False |
| **`` (empty)** | **True** | False |
| **`2`** | **True** | False |

Every disagreement is a way someone writes *off*, and `export INCLUDE_BUSHY=`
meant **on**.

### The consequence is asymmetric, which is what makes it a bug

`app.main` calls `_apply_settings` at import, which pushes settings' answer back
onto `planner`. So the **server** gets `False` — the intended meaning.

The CLI tools never call it. `app.collect_data`, `app.benchmark` and
`app.autotune` see only `planner`'s own import-time read, and get `True`.

So `INCLUDE_BUSHY=false` collects a family the server does not serve. That is
2.56's train/serve gap, reachable by writing a boolean the obvious way, with
nothing to indicate it — the collection log simply contains a family that never
appears at serving, which is precisely the divergence 2.75 and 2.84 spent this
session closing from the other end.

### The fix

`planner.env_flag` and `planner.TRUTHY`, used by both. Nine values now agree and
the tenth (`2`) is false on both sides rather than true on one.

The narrower point is that this was not a *disagreement about a default* — those
match, and `tests/test_settings_defaults.py` was written an hour earlier to check
exactly that and found nothing. It was a disagreement about how to *read* the
value, one level below where anybody looked. The test that found it was written
because the first one passed too easily.

### The pattern, again

That makes it the seventh pair in this section that had to agree and did not:
family lists (2.75), the two regression queries (2.77), query identity (2.78),
`_servable` across three scorers (2.76), the policy list across eight files
(2.71 and the frontend), outcome names across three (2.88's neighbours), and now
a boolean parser.

The generalisable version is not "check your pairs". It is that **a value with
two readers has two meanings until something forces otherwise**, and the forcing
has to be a shared function rather than a shared intention — every one of these
had a comment somewhere asserting the two were kept in step.

## 2.90 The module that documented the policy was the one that broke it

2.89 was two parsers for one boolean. This is the same shape one type over, and
it is worse, because here the policy was written down and the module that wrote
it down is the module that violated it.

`settings._from_env` says it plainly:

```python
    except ValueError:
        # A malformed environment variable should not stop the server booting;
        # fall back to the declared default.
```

That is the right call for input nobody is watching. A typo in a `.env` file or
a stray unit (`MIN_GAIN_MS=5ms`) should degrade to the default and log, not take
the process down at boot on a machine with no operator attached.

`learned` parsed six floats with a bare `float(os.getenv(...))`, and **five of
them ran at import**: three module constants (`DISAGREEMENT_CAUTION`,
`DEFAULT_MIN_RELATIVE_GAIN`, `DEFAULT_MAX_REGRESSION_RATIO`) and two signature
defaults (`risk_lambda`, `min_gain_ms`), which Python evaluates once when the
`def` executes, not per call.

So the failure happened before `_from_env` could run:

```
$ RISK_LAMBDA=abc python -m app.benchmark --help
ValueError: could not convert string to float: 'abc'
```

Not a bad benchmark run — no benchmark at all, and the same for the server,
because `settings` imports `learned`. The module holding the fallback could not
itself be imported to apply it. Every tool in the project died on one character.

### What made it invisible

A signature default is the same trap as 2.35's mutable default, in the readable
direction. `min_gain_ms: float = float(os.getenv("MIN_GAIN_MS", "50"))` reads as
"look this up when called" and means "look it up once, at import". Nothing in
the call site, the tests, or the type annotation distinguishes the two. The unit
tests all constructed `LearnedOptimizer` with explicit arguments, so the default
expression was never the thing under test — and it could not be, since by the
time a test ran, it had already been evaluated.

The reach came from the import graph rather than from the code: `settings`
imports `learned` for `DEFAULT_SELECTION_POLICY` and `RANKER_POLICIES` — a
dependency added for good reasons in 2.43 and again this session — so one
module's boot-time parse became every entrypoint's boot-time parse.

### The fix, and the line it does not cross

`learned.env_float(name, default)` — warns and falls back, matching
`_from_env`'s policy value for value on `abc`, `1,5`, `""`, `2ms` and `None`.

What it deliberately does **not** do is relax the API. `settings._coerce` still
raises on a bad `confidence_z` from a settings POST, and
`tests/test_env_parsing.py::TestApiInputStillRejects` pins that. The two are not
inconsistent: environment input is read at boot with nobody watching, so it
degrades; API input was typed by a person a moment ago into a form that can show
them an error, so it rejects. The distinction is the point, and writing it as
one rule in either direction would be the actual bug.

### The test that had to be a subprocess

The first version of the test used `importlib.reload(learned)`, and turned 14
unrelated tests red. Reloading the module mints a **new** `_UNSET` sentinel
object, and every optimizer already holding a reference to the old one stops
matching `confidence_z is _UNSET` — so the sentinel falls through as if it were
a value and the gate computes `self.confidence_z * combined_sigma`:

```
TypeError: unsupported operand type(s) for *: 'object' and 'float'
```

An identity sentinel is fine until two copies of it exist, which is the same
"two readers, one value" shape as 2.89, arrived at from inside my own test. The
fix is that import-time behaviour is only testable in a fresh interpreter, so
the test now shells out and asserts on the exit code.

### The audit, and a correction to it

Written first as "the remaining reads worth auditing are `GUARD_MIN_OBSERVATIONS`
and `DECISION_CACHE_SECONDS`". Both are wrong: they have no bare parse anywhere.
They are declared in `settings.FIELDS` and reach the code only through
`_from_env`, which is the module that had the policy right all along.

What the audit actually found was seven more, and the two I had named were not
among them:

| module | variable | parsed at |
|---|---|---|
| `app/db.py` | `DB_POOL_MIN`, `DB_POOL_MAX` | import |
| `app/main.py` | `AUTO_RETRAIN_SECONDS`, `ANALYZE_EXECUTE_LIMIT` | import |
| `app/optimizer/planner.py` | `RUNTIME_TIMEOUT_FACTOR`, `SHADOW_SAMPLE_RATE` | import |
| `app/train.py` | `CONFORMAL_ALPHA` | import |

Four integers and three floats, every one at import, and `app/db.py` is the
sharpest of them. That module already carries a long comment explaining that it
builds the connection pool lazily *precisely so that importing it does not
require a running Postgres* — and then parsed two integers at import, which
reintroduces the import-time crash it was written to avoid, from a direction the
comment does not cover.

### One module, not three helpers

The fix is `app/env.py`: `TRUTHY`, `env_flag`, `env_float`, `env_int`, importing
nothing from `app` so that `db` — the bottom of the graph — can use it.
`planner` and `learned` re-export the names they used to define.

That structure is the finding applied to itself. Left alone, this session would
have produced `planner.env_flag`, `learned.env_float` and a third helper
wherever the integers landed: three parsers in three modules, which is precisely
what 2.89 was. `tests/test_env_parsing.py` now asserts
`planner.env_flag is env.env_flag` — the same object, not two that agree today —
and walks the AST of every module in the package for a `float(os.getenv(...))`
or a second truthy set.

The AST matters. The first version grepped source text and was failed by
`planner.py`'s own comment *narrating* the old set, which is 2.84's mistake
repeated: a rule about code, asked of prose. Twice now the instrument has been
the defect, and both times inside a test written to catch a defect of exactly
that shape.


## 2.91 The compiler warned about the wrong half

Found by the test written for 2.90 rather than by looking: the AST walk over
every module in the package emitted a `DeprecationWarning` on the way past
`app/dbadvisor/query_stats.py`, which had nothing to do with what it was
checking.

The line is the utility-statement filter in the `pg_stat_statements` query:

```sql
-- Utility statements carry no plan and no lesson.
AND query !~* '^\s*(BEGIN|COMMIT|ROLLBACK|SET|SHOW|LOAD|DISCARD|ANALYZE|VACUUM)\b'
```

inside an f-string that was not raw. Python says:

```
SyntaxWarning: invalid escape sequence '\s'
```

and it is pointing at the harmless half. `\s` is *not* a Python escape, so it
survives untouched and Postgres receives the character class it was meant to
get. `\b` **is** a Python escape — backspace, `chr(8)` — so it does not
survive:

```
'^\s*(BEGIN|COMMIT|ROLLBACK|SET|SHOW|LOAD|DISCARD|ANALYZE|VACUUM)\x08'
```

The pattern asks for a control character immediately after the keyword. No
query text contains one, so it never matches, so `!~*` is always true, so the
clause has never excluded a row. The comment above it describes an intent the
code has never carried out.

### The interesting part

The obvious fix is what the warning asks for: make the string raw. It changes
nothing.

| pattern | `BEGIN` | `COMMIT` | `SET work_mem` | `BEGINNING_OF_TIME` |
|---|---|---|---|---|
| as shipped (`\b` to `chr(8)`) | kept | kept | kept | kept |
| raw string, still `\b` | kept | kept | kept | kept |
| raw string, `\y` | **excluded** | **excluded** | **excluded** | kept |

Measured on the live server. A POSIX ARE reads `\b` as backspace too —
Postgres spells the word boundary `\y`, and `\b` is listed in its manual
under the same escapes Python has. So the string being non-raw was not the bug;
it was a second, independent bug that happened to sit on the same character and
that produced the same wrong behaviour, which is why fixing one of them looks
like a fix and is not.

Two layers with the same syntax and different meanings, and a warning that
fires on the layer where nothing is wrong. Had the string been raw from the
start there would have been no warning at all, and the filter would still have
been dead.

### What it cost, and what it says

Nothing measured — `query_stats` is an advisory panel, so the effect was
`BEGIN`, `COMMIT` and `SET` competing for slots in a "heaviest queries" list
against real statements, and a few findings computed over a population that
included them. It never touched the optimizer.

The transferable part is the audit question, which is new and is not the "two
readers" shape the rest of this section keeps finding:

> When a string is written in one language and executed in another, **every
> escape has two readings, and a warning from the outer language tells you
> nothing about the inner one.**

The same applies to the `LIKE` patterns elsewhere in this file (`'%%pg_stat_statements%%'`),
which are fine, and it is the reason the regression test in
`tests/test_query_stats.py` asserts on the *statement text the module actually
sends* rather than on a pattern retyped in the test — a retyped pattern would
have been retyped correctly and passed against the broken code.

Ruff has a rule for the Python half, `D301`, and it was already enabled. It
covers docstrings only, so it never saw this line — though it did fire on the
docstrings of the test I wrote *about* the finding, which needed the `r` prefix
for the same reason.


## 2.92 Five families of collection bought four queries

`PENDING.md`'s top "not started" item, finally actionable: the collection run
with `INCLUDE_MEMOIZE`, `INCLUDE_PARALLEL`, `INCLUDE_JOIN_METHODS`,
`INCLUDE_BUSHY` and `INCLUDE_DISABLE_INDEX` all on had finished, and
`available_families()` reported all seven optional families present in the log
for the first time. So `python -m app.autotune` could at last price them.

I started the sweep and then, while it ran, asked the fourth standing question
about it — **which population is this computed over?** The answer stopped the
sweep.

### Coverage

| family | queries with rows | of 65 |
|---|---|---|
| `resource` | 65 | 100% |
| `scan_method` | 65 | 100% |
| `join_method` | 16 | 25% |
| `memoize` | 16 | 25% |
| `parallel` | 16 | 25% |
| `disable_index` | 16 | 25% |
| `bushy` | 7 | 11% |

The run was stopped partway, which was known. What was not: the 16 are not a
sample. They are every `corr_*` query and nothing else — because `WORKLOAD`
lists its 18 correlated-predicate queries first and is iterated in list order.
An interrupted collection always covers the same prefix of the same workload
family, so re-running it collects the same queries again.

(Not alphabetical, which is what this section first said. `WORKLOAD` groups by
query family and `corr_*` is simply the first group; the mechanism is list
order, and the fix below would be the same either way.)

### The number that matters is not 16

`calibrate.sweep` scores **held-out queries only** — `_held_out_queries`, for
the reason 2.2.1 documents: a gate tuned on queries the model memorised is
tuned against memorised predictions. The bundle holds out 16 of the 65.

Four of those 16 carry any newly collected family.

```
queries in the log                     : 65
held out by the model                  : 16
queries with a new family              : 17
...of those, held out (what is scored) :  4
```

So a 2^7 = 128-space sweep, several hours of it, would have distinguished five
families on the evidence of four queries. Measured directly, restricted to the
queries that have each family, best achievable net improvement with the family
against without:

| family | held-out queries | delta |
|---|---|---|
| `join_method` | 4 | −15 ms |
| `parallel` | 4 | −7 ms |
| `memoize` | 4 | 0 ms |
| `disable_index` | 4 | +2 ms |
| `bushy` | 8 | 0 ms |

Every one of those is inside 2.18's 18.6% run-to-run noise floor on totals this
small. The honest reading is not "these families are worthless" — it is **that
the experiment has not been run yet**, and that the sweep would have reported a
ranking of five families anyway, with a `net_improvement_ms` to the millisecond
and no indication that four queries produced it.

### Two different mistakes, and only one of them was mine to make twice

The dilution — a family that can only act on 16 of 65 queries scoring its
effect over all 65 — is not a bug. Every arm scores the same queries because
the always-on families cover all of them, and a query where the family cannot
act genuinely contributes nothing, which is what the server would do.

The selection is the bug: the collection budget was spent in list order, and
list order is uncorrelated with which queries the sweep can score. Spending it
on 16 queries of which 4 are scoreable is a 75% waste that nothing reports,
because "16 of 65 collected" and "4 of 16 held out" live in different modules
and were never multiplied together.

### The fix is a flag

`app.collect_data --held-out` reads `held_out_queries` from the bundle and
collects exactly those; `--only a,b,c` names them by hand. Both refuse an empty
selection, and `--only` refuses an unknown id rather than collecting nothing and
exiting 0 — a run that reports a total of zero and succeeds is this project's
entire bug class in one line.

The re-collection is running as of this section. What it changes is not the
answer but whether there is one: 16 held-out queries with all five families
instead of four.

### The question this adds

Not a new one, but a sharper edge on the fourth. "Which population is this
computed over?" has an operational corollary when the answer is *collected*
rather than given:

> **The budget and the population must be the same choice.** A measurement
> spends its cost on one set of things and draws its conclusion from another,
> and nothing checks that those two sets are related unless someone multiplies
> the fractions.


## 2.93 The fix for 2.92 deleted a family nobody asked about

2.92 added `app.collect_data --held-out` so a collection run spends its budget
on the queries `calibrate.sweep` can actually score. It worked. Four families
went from 4 held-out queries to 16:

| family | before | after |
|---|---|---|
| `join_method` | 4/16 | **16/16** |
| `memoize` | 4/16 | **16/16** |
| `parallel` | 4/16 | **16/16** |
| `disable_index` | 4/16 | **16/16** |
| `bushy` | 4/16 | 8/16 |
| `scan_method` | 16/16 | **0/16** |

The last row is the finding. `scan_method` had rows on all 65 queries before
this run and has none on any held-out query after it. Nothing was deleted from
the log; 43,327 rows are still there. What changed is which of them count.

### Why

`calibrate._grouped_rows` keeps only the most recent `created_at` per query,
and its docstring gives the reason at length: the log accumulates across every
collection the database has ever seen, the same native query has ranged from
2.4 ms to 13.7 ms between runs, and replaying all of history compares a
candidate measured in one run against a native measured in another. The
all-history replay reported every gate setting losing 15--25% while a live
benchmark of the same gate captured 40.8%. Keeping one run is correct and it is
load-bearing.

The rule is **per query**. Families live inside a query's row set, so
superseding a query supersedes every family in it. My re-collection ran with
`INCLUDE_JOIN_METHODS`, `INCLUDE_MEMOIZE`, `INCLUDE_PARALLEL`, `INCLUDE_BUSHY`
and `INCLUDE_DISABLE_INDEX` set, and without `INCLUDE_SCAN_METHODS`, because I
was thinking about the five families that were unscored and not about the two
that were fine. Those sixteen queries now have a latest run with no
`scan_method` candidates in it, so the family is invisible to the sweep.

`available_families()` derives the sweep axis from the data, which was 2.75's
fix. It would have reported `scan_method` as *not scored*, the sweep would have
run six arms instead of seven, and the output would have looked exactly like a
successful run. The wording is what makes it invisible: *not scored* reads as
"never collected", and this family had been collected on all 65 queries an hour
earlier.

### The shape

This is not the section's usual defect. Nothing has two readers here, and no
two components disagree. One rule is applied at one granularity and quietly
inherited by everything nested inside it:

> **A conflict-resolution rule chosen at one granularity applies to everything
> below it.** "Latest run wins" was a decision about *runs*, taken to stop
> comparing machine states. It is also, unavoidably and unmentioned, a decision
> about *families*, and it makes a narrow collection destructive rather than
> additive.

The two properties that made it costly are worth separating. Being *narrower*
than the previous run is what deleted coverage. Being *targeted* -- the thing
2.92 added on purpose -- is what concentrated the damage on exactly the sixteen
queries the sweep reads.

### The fix

Collect with every family the axis knows about, not merely the ones under
investigation:

```
INCLUDE_JOIN_METHODS=1 INCLUDE_MEMOIZE=1 INCLUDE_PARALLEL=1 INCLUDE_BUSHY=1 INCLUDE_DISABLE_INDEX=1 INCLUDE_SCAN_METHODS=1 python -m app.collect_data --held-out
```

and `collect_data` now warns before it starts when the flags it was given would
produce a narrower family set than the latest run already in the log for those
queries. The warning names the families about to disappear. It does not refuse,
because a deliberately narrow collection is a legitimate thing to want; it
refuses only to let it happen silently.

### A note on how this was caught

By the arithmetic disagreeing with itself. `_load_rows()` said `scan_method`
covered 65 queries and `_grouped_rows()` said 33, of which none were held out,
and both were being used in the same session to answer the same question. That
is 2.70's fourth question -- *which population is this computed over?* -- and
the answer here was that one function reads all of history and the other reads
the latest run, which is exactly the difference that mattered.


## 2.94 Seven families priced, and the answer is none of them

`PENDING.md` has carried "collect the unscored families, then price them" since
2.75. Three collection runs and two defects later (2.92, 2.93) the axis was
finally complete: all seven optional families present on the held-out queries
the sweep actually scores. `app.autotune --policy pairwise_rank` swept all
$2^7 = 128$ action spaces under the served policy.

**Thirteen of the 128 produced a gate that clears the safety bounds.** Here they
are, in full:

| extra families | z | min gain | deviates | regress | worst | net |
|---|---|---|---|---|---|---|
| *none* | 1.00 | 0.00 | 94% | 20% | 1.15x | **8.7%** |
| `bushy` | 1.00 | 0.00 | 94% | 20% | 1.15x | **8.7%** |
| `memoize` | 1.50 | 0.05 | 75% | 17% | 1.15x | 8.1% |
| `bushy,memoize` | 1.50 | 0.05 | 75% | 17% | 1.15x | 8.1% |
| `resource` | 2.00 | 0.05 | 62% | 20% | 1.15x | 6.1% |
| `memoize,resource` | 2.00 | 0.05 | 62% | 20% | 1.15x | 6.1% |
| `bushy,resource` | 2.00 | 0.05 | 62% | 20% | 1.15x | 6.1% |
| `bushy,memoize,resource` | 2.00 | 0.05 | 62% | 20% | 1.15x | 6.1% |
| `join_method` | 2.00 | 0.05 | 62% | 20% | 1.00x | 6.0% |
| `join_method,memoize` | 2.00 | 0.05 | 62% | 20% | 1.00x | 6.0% |
| `bushy,join_method` | 2.00 | 0.05 | 62% | 20% | 1.00x | 6.0% |
| `bushy,join_method,memoize` | 2.00 | 0.05 | 62% | 20% | 1.00x | 6.0% |

The winning action space is `{join_order, operator_toggle, cardinality}` --
the three that are always on. **Not one optional family earns its place**, and
the ordering is monotone downward as families are added: 8.7% with none, 8.1%
with `memoize`, 6.1% with `resource`, 6.0% with `join_method`.

`scan_method`, `parallel` and `disable_index` appear in none of the thirteen.
Every space containing them failed the regression bounds outright.

### What is resolvable here and what is not

The spread across all thirteen survivors is 8.7% to 6.0%, which is **2.7
points**. 2.32 measured two independent 3-run benchmarks of an *identical*
configuration landing **8 points apart**, and the latest run's own floor is 7.7
points. So the entire spread of this table is roughly a third of the
instrument's resolution.

That has to be said plainly: **this replay cannot distinguish any of these
thirteen arms from each other.** "Adding `memoize` costs 0.6 points" is not a
finding, it is a number smaller than the noise. The tool says as much about its
own top two, and the same objection applies down the column.

Two things *are* resolvable, because they do not depend on separating small
differences:

1. **`bushy` changes nothing at all.** Every arm containing it is
   *bit-identical* to the same arm without it -- same z, same gain, same
   deviation rate, same regression rate, same worst case, same net. That is not
   a small difference inside noise; it is exact equality across four pairs,
   which means no bushy candidate was ever selected. 2.84 separated `bushy`
   from `join_order` so the sweep could price it. Priced: zero.
2. **Three families cannot clear the safety bounds at all.** `scan_method`,
   `parallel` and `disable_index` survive in none of the 128 arms. Failing a
   bound is a categorical outcome, not a marginal one.

### The result this actually produces

The recommendation is to serve the narrow space, which is what the deployment
already does. Every `INCLUDE_*` flag stays at 0. After three collection runs,
roughly two hours of wall clock and two defects found on the way, the action
space does not move.

That is worth having. 2.75 recorded five families as *not scored*, which is an
open question sitting in a file; this replaces it with an answer. And it lands
where 2.68 already pointed: on six six-table queries carrying 23.6% headroom,
the model was allowed to act twice and lost both times, and **neither losing
pick came from an optional family** -- both were `operator_toggle` and
`cardinality`, always on. The action space was not the binding constraint then
and it is not now.

### The caveat the tool prints itself

> then confirm by execution -- this was a replay, and 2.32 measured two
> identical 3-run benchmarks landing 8 points apart

A replay scores logged rows against a model. It is not an execution, and 2.83
is the section about a replay handing the model candidates the server would
never offer. The +8.7% is a replay number and stays one until
`app.benchmark --protocol paired` says otherwise on the same gate.
