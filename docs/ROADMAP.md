# How it was built

The stages this project went through, in order, and what each one added. Useful
if you want to understand *why* the system is shaped the way it is -- most of
the design came from a measurement that forced it, not from a plan written up
front.

`ARCHITECTURE.md` describes the finished system. This is the path to it.

**Status: all stages complete, plus a large amount beyond the original scope.**
`docs/WRITEUP.md` has the results and `data/job/README.md` covers the JOB/IMDB
work.

## Stage 0 -- Foundation [done]
- Postgres + pg_hint_plan running in Docker
- Synthetic benchmark schema loaded (users / products / orders / order_items)
- Plan extraction (EXPLAIN JSON to structured metrics) working end to end
- Join-order candidates generated with hints
- A placeholder "learned" optimizer that just picks the lowest estimated cost.
  It proves the pipeline works before any real ML exists

**Milestone: run `python -m app.benchmark` and see real baseline and candidate times.**

## Stage 1 -- Data collection [done]
- Workload grown to 25 queries (`backend/app/workload.py`): 2-way, 3-way and
  4-way joins, across a range of selectivities, each one tagged
- Every (query, candidate plan, actual time) triple logged to the
  `plan_execution_log` table (`postgres/init/03_logging.sql`,
  `backend/app/logging_store.py`). This is the training data
- Skew added to the seed data: 100 power users account for 40% of all orders,
  200 popular products account for 50% of all order items (`data/schema.sql`)

## Stage 2 -- Feature engineering [done]
- `backend/app/optimizer/features.py` turns each (query, candidate plan) pair
  into a fixed-length feature vector: join position, selectivity, scan type,
  and join-method counts per table
- Schema-agnostic by design (see "works on any database" below). This step got
  real time budgeted to it in the original plan, and it is what made the
  JOB/IMDB stretch goal possible without a rewrite

## Stage 3 -- Train the model [done]
- LightGBM gradient-boosted trees (`backend/app/train.py`) predicting
  candidate latency. Falls back to scikit-learn's
  `GradientBoostingRegressor` if the LightGBM native library is missing
- Scored against the cost heuristic, native Postgres, **and** an oracle
  (the best candidate in hindsight). The oracle is what makes the other
  numbers readable. Results in `docs/WRITEUP.md` §2
- **Contextual-bandit stretch goal [done]**: a bootstrapped ensemble
  (`backend/app/optimizer/bandit.py`) supports Thompson sampling, the same
  mechanism Bao uses, plus a risk-averse policy driven by ensemble
  uncertainty. LinUCB was not attempted, because it assumes a linear reward
  model and the ensemble gives uncertainty without that assumption

## Stage 4 -- Online integration [done]
- `LearnedOptimizer._select_learned` built. `select()`'s interface did not
  change: the policy and safety additions are constructor arguments plus a new
  `select_plan()`, so existing callers were untouched
- Safety veto: a learned pick costed far above the native plan is dropped
  rather than served. This is Bao's "never much worse than the optimizer you
  replaced" property
- Every `/query/analyze` call and every `app.benchmark` run logs to
  `plan_execution_log`, tagged with which selector produced it.
  `/stats/trend` turns that into "learned vs. native, over time"
- Cold start handled honestly: it falls back to the cost heuristic until a
  model is trained

## Stage 5 -- Dashboard and writeup [done]
- React (Vite) + Recharts dashboard (`frontend/`): paste a query and see
  PostgreSQL's plan against the chosen one, every candidate measured, why the
  decision went the way it did, and the history so far.
  `docs/DASHBOARD.md` explains each panel, `docs/METRICS.md` explains each
  number
- `docs/WRITEUP.md`: how this compares to published work (Neo, Bao, native
  Postgres), the results, and the things that turned out to matter most

## Beyond the original scope [done]

Everything below came from a measurement rather than a feature list -- something
in the data said the system needed it. `docs/WRITEUP.md` has the evidence for
each.

- **Self-learning loop** (`app/retrain.py`): retrains on accumulated feedback
  and deploys a new model **only if it clearly beats the one running**, scored
  on the same held-out set. The margin exists because offline scores here are
  noisy enough that deploying on any improvement would send the model on a
  random walk.
- **Versioned models with rollback** (`app/model_store.py`): unattended
  retraining is only safe if a bad deployment can be undone.
- **Per-query regression guard** (`app/optimizer/regression_guard.py`): blocks
  the learned path for queries with a measured history of being slower than
  native. This targets the per-query instability that recent research
  identifies as the main barrier to deploying learned optimizers.
- **Pairwise learning-to-rank** (`app/optimizer/ranker.py`): Lero-style.
  Predicting *which plan is faster* is a much easier problem than predicting
  latency, and that matters when prediction error is as large as the spread
  between candidates.
- **Plan-tree structural encoding** (`app/optimizer/plan_tree.py`): a cheap
  stand-in for the tree convolution Neo and Bao use.
- **Model health API and dashboard panel**: deployed version, how much
  feedback is waiting, which queries are blocked, and retrain/rollback
  controls.
- **Works on any PostgreSQL database** (`app/onboard.py`,
  `app/schema_graph.py`, `app/workload_generator.py`): discovers the schema,
  works out join links from column names when no foreign keys are declared,
  and generates a workload with filters sampled from the real data. Checked
  with the same command on the synthetic schema and on the 21-table, 74M-row
  JOB/IMDB benchmark.
- **Production inference path** (`app/optimizer/planner.py`): plans every
  candidate with `EXPLAIN` (nothing runs) and executes only the winner. About
  2.7% overhead, against the demo path's N executions per query.
- **Cumulative regret** (`app/optimizer/regret.py`): 0.80x native over 500
  logged decisions. The trajectory, not just an average.
- **Learned cardinality correction** (`app/optimizer/cardinality.py`): learns
  Postgres's own q-error from `Plan Rows` against `Actual Rows`, the root
  cause Leis et al. identified. At join level the correction is fed back
  through `pg_hint_plan`'s `Rows()` hint, so the planner searches with a better
  premise instead of being overridden. Verified to bind against a live planner
  (4 estimated rows becomes 400 under `Rows(o u *100)`); measured to raise the
  oracle ceiling by 0.0% on this schema, and written up as the null result it
  is (`docs/WRITEUP.md` 2.8.1). Disable with `ENABLE_ROWS_CORRECTION=0`.
- **Paired A/B experiments with confidence intervals** (`app/experiment.py`):
  interleaved arms, bootstrap intervals and a sign test, so a comparison is
  reported as *unresolved* when the interval spans zero instead of being read
  off three runs. `app.benchmark` returning nothing was the reason the sample
  size in §2.4 was 3.
- **Integration tests and CI** (`tests/test_integration_hints.py`,
  `.github/workflows/ci.yml`): the unit suite runs with no database; a second
  job builds Postgres with pg_hint_plan and asserts against a live planner that
  hints actually bind. These are the tests that would have caught §2.0, whose
  absence let 200 passing tests sit on top of a pipeline doing nothing.
- **Honest dashboard reporting** (`app/stats.py`): every before-and-after
  figure is a matched pair from the same run. The panel previously divided two
  averages taken over different sets of queries and reported a 97% win that
  was not real. `docs/METRICS.md` §2 works through it.

## Stretch goals
- **Join method selection [done]**: `generate_join_method_candidates`
  (`backend/app/optimizer/hints.py`) goes beyond join order to force
  `HashJoin` / `NestLoop` / `MergeJoin` per join-order prefix
- **Real Join Order Benchmark (JOB/IMDB) dataset [attempted, see
  `data/job/README.md` for the outcome]**: a loader script
  (`data/job/load_job.sh`) downloads and imports the real 21-table IMDB
  dataset and 113 JOB queries into a second database. The join-order and
  join-method pipeline needed zero code changes to point at it, thanks to the
  stage 2 schema-introspection work


---

# What to build next

Ranked by what the project's own measurements say is binding, not by how
interesting it sounds. Each entry names the evidence that puts it where it is.

## Tier 1 — attacks a measured, binding constraint

### 0. Re-establish every headline number under the served policy

**Nothing below this line can be trusted until this is done.** 2.43 found that
`pairwise_rank` -- what the container serves -- was named in eight places and
defaulted to `greedy` in seven of them, including `benchmark.run`'s own
signature. Every benchmark figure this project has published was therefore
measured under a policy nobody runs, and the two are not close: on 15 held-out
queries `greedy` lands at 104.1% of native and `pairwise_rank` at 95.1%.

The defaults are fixed and consolidated. What remains is arithmetic:

1. re-run `app.benchmark --protocol compare` and replace the headline;
2. re-run promotion (`app.retrain --force`) with the corrected yardstick (2.45)
   and policy, then calibrate the promoted bundle and `--apply` the gate;
3. re-read 2.37's eleven 0.0% runs as measurements of `greedy`, which is what
   they were.

**Done, 2026-09-04.** The benchmark now reads **41.3% of available headroom
captured, 45.7% on held-out queries** (2.51), against 0.0% held-out on every run
ever previously measured. The corrections that produced it are 2.42 through 2.50;
nothing changed in the model, the features or the action space.

**Steps 2 and 3 are done.** A model trained on 22,544 servable rows with 15
queries held out was promoted on 2026-09-04 -- on worst case, not mean: the
champion's worst pick was 118.98x the best available plan against the
challenger's 4.68x, while the mean went the other way at -8.2%. `app.autotune`
then chose `pairwise_rank`, no `resource` family, `fixed_z` at z=2.0 and
`min_relative_gain` 0.0, and `models/gate.json` records the policy and action
space it was swept under. Step 1 is the remaining one, and 2.47 is why the
number it produces should now be believed: the replay and the executed path
finally agree, having disagreed by 5.4 points until the safety veto was
modelled.


### 1. Reload JOB into PostgreSQL 18 and re-run it &mdash; **done, 2026-09-05**

`docs/JOB_RESULTS.md` recorded this system capturing **none** of a 75% oracle
gap on the real 21-table IMDB benchmark (native 3703 ms against an oracle of
919 ms on held-out queries). Everything built since &mdash; shape coverage (2.35),
the 59-query workload, the worst-case promotion metric (2.31), the PG18
features, the corrections in 2.42-2.54 &mdash; had never been tested against it.

**Reloaded**: 8.7 GB, 73.0M rows, 22 tables, via `bash data/job/load_job.sh`.
`app.collect_data_job` runs from the host now (`--query-dir`) rather than only
from inside the backend container.

**Re-run**: all 30 of the smallest queries, 1614 raw rows against the original's
194, **8 held out** against its 2. The held-out queries average **3960 ms**
native against the original's 3703 ms &mdash; the same expensive-query regime,
which is what makes it a comparison rather than a different measurement.

| selector | avg latency | headroom captured |
|---|---|---|
| native | 3960.2 ms | &mdash; |
| cost heuristic | 2285.3 ms | +42% |
| oracle | 695.6 ms | 100% |
| **`risk_averse`** | **713.7 ms** | **+99.4%** |
| `pairwise_rank` | 722.8 ms | +99.2% |
| `greedy` | 726.9 ms | +99.0% |
| `thompson` | 863.9 ms | +94.8% |

All four policies clear 94% and the top three sit within 0.4 points, so this is
a property of the candidate set and the model rather than of one selection rule.
`risk_averse` leading reproduces 2.5's finding that data-starved JOB rewards
modelling uncertainty over modelling the target.

On three of the eight it picks the **oracle plan exactly** (within 0.1 ms on a
fourth) &mdash; `17c` goes from **21.6 s to 2.0**, 76% of all the saving. The other four regress by **-208 ms**
total against 26,099 ms saved, and three of those had 2.7 ms, 14.4 ms and
*nothing* available respectively. Ungated numbers: the confidence gate exists
for exactly those near-ties.

**What is left**, and it is smaller than what was closed: the benchmark's larger
queries &mdash; JOB goes to 17 tables &mdash; have never been touched, and eight
held-out queries is more than two and still few.

### 1b. Re-collect under the widened action space

2.56 raised the ceiling and nothing has been re-measured against it. Three fixes
landed: `budget_for` scales with join width as well as cost (`17c` goes from 20
candidates to 90), `hint_family` names the 39% of the action space that
classified as `other` and was therefore dropped from every replay, and
`app.collect_data` forwards the same flags and budget as serving instead of
`generate_candidates`' bare defaults.

Confirmed on two queries: `2b`'s available headroom went from **0.6% to 36.7%**
and `2d`'s from **2.5% to 34.1%**, both on a single `SeqScan(...)` hint.

What is left is arithmetic, and it is slow rather than hard:

1. re-collect all 30 JOB queries with `INCLUDE_SCAN_METHODS=1` (the `17*` family
   is twenty minutes each, so budget two to three hours);
2. re-train and compare against the 99.2% recorded above, which is a figure
   against the *old* ceiling;
3. run `app.autotune`, which sweeps `scan_method` as a fourth optional family
   now, to settle whether it earns its planning cost per database;
4. do the same on the synthetic workload and re-run the headline benchmark --
   `17c` measured 80 candidates at 352 ms of planning, 1.63% of its runtime, but
   the synthetic queries are milliseconds and the ratio there is unmeasured.

Step 4 is the one with a real risk attached: widening the search costs planning
time on every query, and the synthetic workload is where that cost is largest
relative to the query.

### 2. Fix the ranking on unseen queries

**This replaced "replace the uncertainty estimate", and the reason is worth
keeping.** That item asserted the gate was the binding constraint, citing 2.37:
the gate declined 20-43% of available headroom on five held-out queries.
Measured, that inference was wrong (2.41). The headroom was measured against
the **oracle** — the best plan in the action space — and the gate does not have
access to the oracle. It has access to the model's ranking.

With the gate fully open on 15 held-out queries, following every `greedy` pick
is net **-2.93%**, half the deviations regress, and the worst is 1.73x.

**Then 2.43 found that `greedy` is not what serves.** Under `pairwise_rank` the
same 15 queries give net **+4.55%**, and the served policy lands at 95.1% of
native against an oracle of 83.6% — **30% of the available headroom captured on
queries the model has never seen.** So the ranking is not broken; the *ensemble*
ranking is, and the ensemble was never the thing being served.

That leaves a real and much better-specified target. Within a query's candidate
set the ensemble's predictions correlate with true latency at Spearman **+0.533**
(median over 15 held-out queries, positive on 14). A random pick costs 163.7% of
native and the ensemble's argmin costs 104.1%, so it removes 60 of the 64 points
between random and native — and then fails, because the bar is not "better than
random" but "better than native", and native sits only 16.4 points above the
oracle. The ensemble is good enough to avoid disasters and not good enough to
separate the top few candidates, which is where all the remaining opportunity
is.

That reframes 2.35's "uniformly inaccurate" result rather than repeating it:
uniform inaccuracy is survivable for a *ranker* and fatal for a *regressor used
as a ranker*. 2.51x typical multiplicative error mostly cancels in a comparison
and does not cancel at all in a prediction. The work is to spend the model's
capacity on comparisons:

- **give the ranker the attention the ensemble has had.** Every diagnostic in
  section 2 — `uncertainty_scale`, the conformal bound, the calibration sweep,
  the promotion score — describes the ensemble. The component that actually
  chooses plans has none of them;
- features that describe the *query*, not only the plan (item 5) — the plan is
  what PostgreSQL believed, and it is wrong in exactly the cases that matter;
- per-join-width models (item 7), for which 2.40 and 2.45 are two independent
  arguments;
- a label less noisy than wall-clock: buffer counts (item 13) do not vary 18.6%
  between identical executions the way latency does (2.18).

The uncertainty estimate is *not* on this list any more. `uncertainty_scale` is
still 3.75 and still means the ensemble spread is not a standard deviation — but
the gate no longer has to believe it is. 2.42 made the conformal bound an axis
of `app.calibrate`'s sweep, so a gate built on the empirical quantile of the
model's own residuals now competes with `mean + z·sigma` on measured terms.
Quantile regression is a fine thing to try and would sharpen that bound; it is
no longer load-bearing.

### 3. Act on the correlated columns instead of planning around them

`dbadvisor/extended_stats.py` already **finds** them — it measured `('SG',
'Singapore')` occurring **22x** more often than per-column statistics predict.
It then reports it and stops.

`CREATE STATISTICS ... (dependencies, mcv)` fixes the estimate at its source.
That helps every query against those columns forever, including ones this
optimizer never sees, and it *shrinks the headroom the optimizer is chasing* —
which is a better outcome than getting better at chasing it.

### 3b. Refit the conformal gate on serving-shaped spreads

`optimizer/conformal.py` uses `normalize=False` on measured grounds: over 2,483
calibration rows the sigma-normalised bound was unusable at every risk level,
because rows where the eight ensemble members agree almost exactly give a
near-zero denominator that dominates the quantile.

Those calibration rows are **training** rows, where the spread is ~0.026. On
held-out queries the spreads run 0.018 to 0.334 (2.46). So the bound is
calibrated where its key input is nearly degenerate and applied where it is not,
and the constant-width interval that results admitted a 305 ms query served at
598 ms -- the whole of conformal's -11.64% under the served policy, against
`fixed_z`'s +3.10%.

The near-zero-denominator failure is a property of the calibration sample, not
of the method. Refit with `normalize=True` on residuals whose spreads resemble
serving, and re-score both bounds through `app.calibrate`'s existing sweep.

## Tier 2 — new capability, grounded in something measured

### 4. Shadow execution

Run the model's pick and native, compare, serve the winner, log both. Every
served query becomes a labelled training pair at zero regression risk. Today the
regression guard learns only from regressions it has already caused, which is an
expensive way to acquire a label.

### 5. Features from the query, not only the plan

The model sees plan trees. Two queries with identical plan shapes and different
predicates are, to it, the same query. Predicate selectivity, literal
cardinality and operator type are all available before planning.

### 6. `pg_overexplain` features

Verified working on 18.6: `Disabled Nodes`, `Parallel Safe`, `Plan Node ID`,
relation OIDs — per-node internals no `EXPLAIN` option reports. Free signal, now
that the extension is installed.

### 7. Per-join-width models

The workload spans 2 to 6 tables (18/13/15/10/3). A six-way join's plan space
does not resemble a two-way's, and one global model has to compromise between
them.

### 8. Incremental training

Retraining is batch, champion/challenger. LightGBM supports `init_model` for
continued training, which would let the model absorb feedback between full
retrains instead of waiting for one.

## Tier 3 — worth doing, less certain

### 9. Widen the action space, now that inference is nearly free

2.39 took inference from 8.27 ms to 0.786 ms and showed the cost is **fixed**,
not per-candidate: 50 candidates cost barely more than 1. `CANDIDATE_BUDGET_TIERS`
was shaped when inference appeared to scale with candidate count. The marginal
candidate now costs about 0.87 ms of `EXPLAIN` and 0.03 ms of inference, so the
budget is optimising against a cost model that no longer holds.

### 10. Re-measure the two switched-off families

`INCLUDE_JOIN_METHODS` and `INCLUDE_BUSHY` both bind correctly since 2.20 and
2.29. 2.27 measured zero wins across 24 queries — before the workload tripled
and before the shape group existed.

### 11. `DisableIndex` for the unused-index advisor

Works on pg_hint_plan 1.8.1 (measured: Index Scan 8.4 -> Seq Scan 4059.0). It
turns "this index has zero recorded scans", which is evidence about the past,
into "here is what dropping it would do to your plans", which is evidence about
the future. Dropping an index is irreversible without a rebuild.

### 12. Parameterised queries and the plan cache

Every workload query uses literals; real traffic is parameterised.
`GENERIC_PLAN` (PG16+) and `plan_cache_mode` are the levers, and
`query_template.py` already exists to give parameterised statements one identity.

### 13. Buffer counts as a training label, not just a diagnostic

`measurement.blocks_verdict` computes them. Latency on this database varies
**18.6%** run to run (2.18); buffer counts barely vary at all. A label with less
noise may be worth more than a label that is exactly what you want to predict.

### 14. Cross-database transfer

Train on `lqo`, serve on JOB. The features are schema-agnostic by construction
(`scan_relations`, `schema_introspection`) and that claim has never been tested
in the direction that would falsify it.

### 15. Teach the regression guard *what* to block &mdash; **done**

The entry described the guard blocking a whole query after three bad
observations, and proposed blocking the hint *family* instead. That was built:
`blocked_families`, `is_family_blocked` and `allowed_hints` exist, are wired into
`planner.plan_query`, and are tested. The entry had gone stale.

What the re-check *did* find is that `app.benchmark` built its candidate list
straight from `candidate_hints` and never applied the filter, so with the guard
on it scored a wider action space than production serves. `--protocol compare`
runs guardless and is unaffected; `--protocol deploy` was overstating. Fixed,
with a test that the filter stays conditional on `use_guard` &mdash; applying it
under `compare` would reintroduce the guard's state into a protocol built to
exclude it (2.30).
