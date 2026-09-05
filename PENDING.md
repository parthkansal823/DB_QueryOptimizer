# Pending work

Everything known to be unfinished, with the measurement that says why it
matters. Ordered by what would change the project's claims most, not by effort.

`docs/ROADMAP.md` is the same list with the reasoning; this is the checklist.
`docs/WRITEUP.md` §4 is the honest list of what is still wrong.

Last updated 2026-09-05, after §2.56.

---

## Blocking — numbers that are currently stale

These are not new features. They are figures the project quotes that were
measured against something that has since changed, and until they are re-run the
documentation overstates or understates the system.

### P1. Re-collect JOB under the widened action space

**Why**: §2.56 raised the ceiling and nothing has been re-measured against it.
The **99.2%** headroom-captured figure in `docs/JOB_RESULTS.md` is correct
against the *old* oracle and an overstatement against the new one. Confirmed on
two queries: `2b`'s available headroom went from **0.6% → 36.7%** and `2d`'s from
**2.5% → 34.1%**, both on a single `SeqScan(...)` hint that was never collected.

```bash
# ~2-3 hours: the 17* family is ~20 min each
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/job \
INCLUDE_SCAN_METHODS=1 python -m app.collect_data_job --n 30 --reps 2
DATABASE_URL=...job python -m app.train --model-out models/plan_selector_job18.pkl \
                                        --eval-out models/eval_job18.json
```

- [~] re-collect all 30 with `INCLUDE_SCAN_METHODS=1` — **stopped after 14 of
      30**, deliberately. The wider budget gives the `17*` family 114 candidates
      instead of 30: `17d` took **64 minutes** and `17e` was still running after
      **2h21m**. Four more `17*` queries would have been 10+ hours, and it was
      blocking P2 — which is the headline number and needs a quiet database for
      clean latencies.

      The question P1 was asked is already answered by the partial data (below).
      What is left is a clean 30-query re-train, and it needs a tighter timeout
      or an overnight run:

      ```bash
      DATABASE_URL=...job INCLUDE_SCAN_METHODS=1         python -m app.collect_data_job --n 30 --reps 2 --timeout-ms 30000
      ```

      30 s is a real trade rather than a free speed-up: `17c` runs 21.6 s
      natively, so that cap admits only candidates within ~1.4x of native and
      the model stops seeing moderately-bad plans for those queries. The
      narrow-space rows from the earlier complete run still carry them.

**Partial result, and it already settles the question P1 was asked.** Oracle over
every row collected so far, old narrow-space and new wide-space together:

| query | old oracle | new oracle | old headroom | new headroom |
|---|---|---|---|---|
| `2b` | 460.9 | **311.1** | 0.6% | **34.0%** |
| `2d` | 571.7 | **395.5** | 2.5% | **40.0%** |
| `5b` | 83.3 *(worse than native)* | **68.2** | none | **9.4%** |
| `2c` | — | 16.4 | — | 96.5% |
| `6b` | — | 85.8 | — | 79.0% |

Across the 23 queries with data, **74.6%** of the total latency is now reachable.

`5b` is the one that matters most for honesty: §2.55 describes it as a query
where "no candidate in the action space beats PostgreSQL, so any deviation is a
loss by construction". That was true of the narrow action space and is now false
— it has 9.4% available. The same correction as `2b` and `2d`, on the query I
used to argue the regressions were not the model's fault.
- [ ] re-train and compare against 99.2%
- [ ] update `docs/JOB_RESULTS.md`, `docs/WRITEUP.md` §2.55, `README.md`

### P2. Re-run the synthetic headline under the widened action space

**Why**: same reason, and this one carries a **risk** rather than just an update.
Widening costs planning time on every query it touches. `17c` measured 80
candidates at 352 ms — 1.63% of a 21.6-second query. The synthetic queries run in
27–344 ms and the ratio there is unmeasured.

`budget_for` now declines to widen anything measured under 500 ms, so a query
with history is unaffected; **a query without history still widens**, and the
synthetic workload's first run after a reset is exactly that case.

- [x] re-collect the workload — **4,499 rows across 65 queries** under the wide
      space, with `INCLUDE_SCAN_METHODS=1`
- [x] retrain — best train-time numbers the project has recorded: error
      **1.27x** (was 1.31x), `pairwise_rank` capturing **78.8%** of oracle
      headroom (was 62.7%), 67 features, 16 held out
- [x] **and the retrain was rejected**, on the first genuinely uncontaminated
      promotion comparison the project has had (§2.45's fix: six queries,
      `champion_trained_on: 0, challenger_trained_on: 0`).

      On `shape_group_by_category` — 293 candidates, native 141.4 ms, best
      **55.0 ms** — the champion picks `SeqScan(o)` at 57.3 ms and the
      challenger picks `Set(enable_nestloop off)` at **170.1 ms**, predicting
      **0.610x** for a plan that ran at 1.20x native. The gate reads the
      prediction, so a confident wrong prediction is precisely what it cannot
      decline. Written up as §2.63.
- [x] `python -m app.benchmark --protocol compare --repeat 3`, on the 65-query
      workload with the deployed model:

      | | 59 queries (before) | 65 queries (now) |
      |---|---|---|
      | held-out (never trained on) | 45.7% | **43.0%** [23.8, 48.5] |
      | curated (not selected) | 33.0% | **26.0%** [23.6, 42.2] |
      | all queries | 41.3% | **40.8%** [38.3, 46.0] |

- [x] confirm no regression — **every change is inside the noise**. The
      benchmark's own floor for this sample is **7.7 points** and the largest
      move is 7.0. Nothing here is a result in either direction.

**The selection effect widened from +8.3 to +14.8 points**, and that is the six
new queries doing their job rather than a regression. Three of them have no
headroom by construction (including the control), they were added *without* a
headroom filter, and they land in the curated group — so the unbiased number
went down while the filtered-for-headroom `gen_*` group did not. The gap between
the two is a better estimate of the bias than it was.

- [ ] check whether the model *deviates and loses* on the new zero-headroom
      queries — a query with no headroom contributes nothing to either side of
      the capture ratio unless the model regresses on it, so a 7-point drop in
      the curated figure is worth attributing rather than assuming

**The open question this leaves.** The wider space raised that query's ceiling
from 141.4 ms to 55.0 ms (**61% available**) and introduced a 170 ms trap the
model prefers. More options is more upside *and* more ways to be wrong, and which
dominates depends on model accuracy rather than on the action space. At 1.27x
typical error the model reaches part of the new ceiling and not all of it.

- [ ] decide whether `INCLUDE_SCAN_METHODS` should default on once a model can
      exploit it safely — currently the ceiling moved and the model cannot
      follow

### P3. Settle whether `scan_method` earns its place, per database

**Why**: it wins on JOB (`SeqScan` is the best plan on `2b` and `2d`) and is
**off by default**. `app.autotune` sweeps it as a fourth optional family now, so
this is one command — but it can only measure a family the log contains, which
is P1's output.

```bash
python -m app.autotune            # after P1
python -m app.autotune --apply    # if it wins
```

- [ ] run after P1 and P2, on both databases
- [ ] `INCLUDE_SCAN_METHODS` default follows the measurement, not the habit

---

## High value — attacks a measured, binding constraint

### P4. Fix the ranking on unseen queries

**Why**: the model captures 45.7% of held-out headroom on synthetic, so more than
half is still on the table. §2.35 measured the model *uniformly* inaccurate
rather than overfitted — 2.51x typical multiplicative error on trained queries,
2.66x on held-out — and 22,544 rows have not moved it. That points at the
features and the label, not the sample size.

§2.43 sharpens it: uniform error mostly cancels in a *comparison* and not at all
in a *prediction*, which is why `pairwise_rank` beats every point-prediction
policy by 36 points. The work is to spend the model's capacity on comparisons.

- [ ] **give the ranker the diagnostics the ensemble has.** `uncertainty_scale`,
      the conformal bound, the calibration sweep, the promotion score — every
      one describes the ensemble. The component that actually chooses plans has
      none of them
- [ ] features from the *query*, not only the plan (P9) — the plan is what
      PostgreSQL believed, and it is wrong in exactly the cases that matter
- [ ] per-join-width models (P10) — §2.40 and §2.45 are two independent arguments
- [ ] a label less noisy than wall clock: buffer counts (P14) do not vary 18.6%
      between identical executions the way latency does (§2.18)

### P5. Act on the correlated columns instead of planning around them

**Why**: `dbadvisor/extended_stats.py` already **finds** them — it measured
`('SG', 'Singapore')` occurring **22x** more often than per-column statistics
predict. It reports this and stops.

`CREATE STATISTICS ... (dependencies, mcv)` fixes the estimate at its source.
That helps every query against those columns forever, including ones this
optimizer never sees, and it *shrinks the headroom the optimizer is chasing* —
a better outcome than getting better at chasing it.

- [x] generate the `CREATE STATISTICS` statements — already existed
      (`_definition`, `recommend`, `verify_recommendation`)
- [x] **fixed: the advisor could not see one of the two designed traps.**
      `correlated_pairs` took the first 24 of 58 candidate pairs
      *alphabetically*, so `suppliers` and `users` were never evaluated —
      including `users(city, country)`, measured at 63.1% headroom in §2.9. It
      passes easily when evaluated (q-error 1.89 → 1.00). Now round-robin
      across tables, largest first; recommendations went 2 → 3
- [x] **measured: the headroom falls, and by more than expected.** Three rounds
      alternating on/off, on the eight queries touching the correlated columns:

      | arm | native | oracle | headroom |
      |---|---|---|---|
      | statistics ON | 595.9 | 567.4 | **0.4%** [0.2, 4.8] |
      | statistics OFF | 602.0 | 481.4 | **18.6%** [14.2, 20.0] |

      Ranges **disjoint**. Native unchanged — the mechanism is that
      PostgreSQL's own plan is now already good, so the oracle rises to meet
      it. Three statements and one `ANALYZE` remove **98%** of the opportunity.
      Written up as §2.57.

**Left off deliberately.** The deployed model and the 41.3%/45.7% benchmark were
both measured without them, so turning them on needs a re-collect (P2). The
decision is now a documented trade rather than an oversight:

- [ ] decide whether `data/schema.sql` ships with the statistics created
- [ ] if yes: re-collect, re-train, and expect the headline to **fall** — that
      is the metric working, not breaking

### ~~P6. Refit the conformal gate on serving-shaped spreads~~ — **measured and rejected**

The premise came from §2.46: `normalize=False` was chosen because near-zero
denominators made the normalised bound explode, so floor sigma at a percentile
of the observed spreads and it should become usable.

The floor is built (`sigma_floor_percentile`, fitted at calibration and applied
identically at serving — conformal coverage requires the same score function in
both places). Measured on 1,525 rows across 59 queries, query-level split:

| variant | q | held-out coverage | target |
|---|---|---|---|
| **absolute (current default)** | 0.2876 | **90.6%** | 90% |
| normalised, `MIN_SIGMA` only | 5.6073 | 83.8% | 90% |
| normalised, p25 sigma floor | 5.2700 | 83.8% | 90% |
| normalised, p50 sigma floor | 4.0825 | 85.8% | 90% |

**The diagnosis was wrong.** `rows with sigma < 1e-4: 0 (0.0%)` — the smallest
sigma in the whole sample is 5e-3, five thousand times `MIN_SIGMA`. There are no
degenerate denominators to floor. The real reason normalised is worse is
§2.4.5's: sigma understates error by **3.75x** and not uniformly, so dividing by
it adds variation rather than removing it.

The floor was kept anyway — inert for absolute gates, cheap, and the right guard
for a future ensemble that *does* agree too tightly. Written up as §2.59.

**Coverage is split-dependent and worth re-checking**, not a standing failure:
this split gives the absolute bound 90.6% against `eval_results.json`'s 0.755 on
train's own held-out list.

- [ ] decide whether `app.train` should report coverage over several splits
      rather than one, since a single query-level split moves it by 15 points

---

## Known-wrong, cause understood, fix not attempted

### ~~P7. The workload is partly selected to be winnable~~ — **resolved by ordering**

27 of the 59 queries were generated and *kept* because the best plan in the
served action space beat native by ≥8%. The effect was already measured (curated
33.0% against all-queries 41.3%) and the biased figure was still printed first,
which is how a project quotes its most flattering number without meaning to.

- [x] **the summary now leads with the figures nobody selected for**, and names
      the gap instead of leaving it to subtraction:

      ```
      held-out (never trained on)      45.7%   range 19.3% to 50.4%
      curated (not selected)           33.0%   range 20.8% to 34.2%
      all queries                      41.3%   range 38.9% to 43.0%

      the +8.3 point gap between the last two lines is the selection effect:
      27 of the workload's queries were kept for having headroom, and the
      curated group was not.
      ```
- [x] tests assert the ordering and that the note is absent when there is no
      generated group to compare against — a zero-point gap would imply an
      effect had been measured

**Not done, and a real option rather than a leftover**: generating a second
workload with no headroom filter would give an unbiased estimate over *generated*
queries too, rather than only over the 32 hand-written ones. The curated group is
unbiased but small and hand-chosen, so it has its own selection story.

- [ ] generate a workload with the ≥8% filter removed, and compare

### P8. Measurement cannot resolve the differences being argued about

Three compounding problems, all measured: repeated executions of the same plan
vary **18.6%** (§2.18); the benchmark mutates the guard's state, so two runs of
an identical gate gave 15.6% and 7.2% (§2.30); two independent 3-run
measurements of an identical configuration gave medians **8 points apart**
(§2.32). The 2026-09-04 held-out figure spans **19.3 to 50.4**.

The tooling respects this now (`MIN_POLICY_SWITCH_POINTS`, median-of-draws), but
the instrument itself has not improved.

- [ ] more repetitions, or a paired design that cancels the between-run drift
- [ ] until then: no claim resting on a difference under ~4 points

---

## New capability, grounded in something measured

### ~~P9. Features from the query, not only the plan~~ — **built, needs a retrain**

Every feature was read off PostgreSQL's chosen plan — which is what PostgreSQL
*believed*, and it is wrong in exactly the cases with headroom.

**Measured before building it.** The specific hypothesis: PostgreSQL multiplies
the selectivities of `a = x AND b = y` on the same table as though the columns
were unrelated, so the count of same-table equality predicates should predict
where the estimate is wrong. Across 59 workload queries, median available
headroom:

| same-table equalities | queries | median headroom |
|---|---|---|
| 0 | 30 | 16.9% |
| 1 | 3 | 25.3% |
| **2 or more** | 26 | **31.4%** |

Monotone, and nearly double at the top.

- [x] `predicate_shape()` reads the WHERE clause for
      `max_same_table_equalities`, `n_equality_predicates`,
      `n_range_predicates`. Join keys are excluded — `o.user_id = u.id`
      estimates a join, not a filter on one table
- [x] `sql_text` threaded through **both** sides. Training attaches it in
      `_row_to_candidate`, serving in `plan_query`, and the benchmark in `run()`;
      a test asserts all three, because a feature populated on one side and
      absent on the other is the skew shape of §2.43/§2.47/§2.50/§2.56
- [ ] retrain so the model can use it (folds into P2), then check whether the
      splits actually land on it — `num_joins` has been available and unused all
      along, so availability is not usefulness

### ~~P10. Per-join-width models~~ — **measured infeasible; the workload was the fix**

§2.40: one 6-table query carried 664 ms of a 682 ms promotion gap, and only 3 of
59 workload queries were 6-table.

**Separate models per width are not available on this data.** Measured:

| width | rows | queries |
|---|---|---|
| 2 | 432 | 18 |
| 3 | 370 | 13 |
| 4 | 474 | 15 |
| 5 | 290 | 10 |
| **6** | **96** | **3** |

96 rows across 3 queries is far below what §2.3 established as usable.

**And the model already has width and ignores it.** Across the eight ensemble
members `num_joins` gets **0** splits and `num_tables` 65 (0.42% of all). Not a
missing feature — a missing sample.

Six tables is the maximum this schema allows, so the fix is more *variety* at
that width, not more width:

- [x] six 6-table queries added, differing in filter placement, selectivity and
      result shape. Width 6 goes **3 → 9**, workload 59 → 65
- [x] **none filtered on having headroom** — measured after the fact, two have
      30.8% and 27.2%, three have none, and the control correctly has none. All
      six stay: a query PostgreSQL already gets right is what teaches the model
      not to deviate, and filtering them out is the §P7 bias
- [ ] re-collect and re-train so the extra width actually reaches the model
      (folds into P2)

### ~~P11. Shadow execution~~ — **done**

Shadow execution already existed, but did the *opposite* thing: it ran native
alongside a served deviation, pairing decisions the guard already allows. The
gap was the one this entry names.

**The ratchet, stated precisely.** The guard blocks a query when three served
executions of the learned path average worse than native. From then on the query
serves native, so `is_chosen AND hint IS NOT NULL` matches nothing new for it and
the learned path accumulates *no further evidence at all*. The block lifts when
the old rows age out of `evidence_since()` — a **timer, not a decision** — and
the query is retried against whatever model is deployed, having learned nothing
in between. If that model is no better it regresses, is blocked, and the cycle
repeats with a period equal to the lookback window.

Now: on a sampled fraction of blocked requests the learned plan is run **after**
the user has been served native, with the same deadline a real deviation gets,
and logged under `SHADOW_SELECTOR`.

- [x] the guard counts those rows as evidence; `app.stats` excludes them — the
      plan was measured, not delivered
- [x] selection goes through `plan_query(..., bypass_block=True)` rather than a
      copy of it. Evidence about a *different* decision procedure is evidence
      about nothing, and this codebase has §2.43, §2.47 and §2.50 to show for it
- [x] the bypassed call does not itself shadow, or one blocked query spawns a
      measurement per level forever
- [x] runs last, so a protected request never waits on it; a failed shadow is
      not a failed request

10 tests, including that the block still blocks when not bypassed.

### ~~P12. `pg_overexplain` features~~ — **measured and rejected**

Nearly free (`DEBUG` costs **+0.8%** on an EXPLAIN, `RANGE_TABLE` +0.1%), so the
cost objection does not apply. It just has nothing usable.

Three of its four candidate fields are **constant across a query's candidates**
— `Subplans Needing Rewind`, `Disabled Nodes` and `Lateral` all 0/59 — and a
field that cannot separate candidates cannot help choose between them. The one
that varies, `Parallel Safe` (56/59), has a median Spearman with latency of
**+0.040**, positive on 32/56, and its strong cases point both directions
(+0.925 and −0.732).

Written up as §2.61, precisely so the next reading of the PG18 release notes
does not spend the same afternoon.

### ~~P13. Incremental training~~ — **measured, not worth doing**

A full retrain from scratch, on the current log:

| | |
|---|---|
| raw rows in the log | 32,052 |
| after dropping disabled plans | 29,121 |
| after median-aggregating repetitions | **2,433** |
| load from the database | **6.1 s** |
| filter + aggregate | 0.2 s |
| **total, end to end** | **11.1 s** |

Eleven seconds, in a background job triggered when enough new rows accumulate.
Incremental training could save at most the ~5 s of model fitting, and would be
genuinely hard to get right: the bootstrapped ensemble's out-of-bag structure —
which the conformal gate and `uncertainty_scale` both depend on — is defined by
which rows each member did *not* see, and that is exactly what continuing
training breaks.

**The aggregation is why it stays cheap.** 32k raw rows collapse to 2,433, so
fitting cost tracks distinct `(query, hint)` pairs rather than executions. Ten
times the data is still seconds.

- [ ] if it ever does matter, the lever is incremental **loading**, not
      incremental training — the database read is 55% of the total and
      `_load_rows` taking a `since` is a far smaller change than continuing a
      bootstrapped ensemble

### ~~P14. Buffer counts as a training label~~ — **measured and rejected**

The premise holds completely and the conclusion does not. Over 27,001 logged
rows (§2.58):

| | median spread across repeats |
|---|---|
| wall-clock latency | **96.4%** |
| buffer count | **0.0%** |

Identical across every repetition on 1,136 of 1,721 `(query, hint)` pairs. But
they rank candidates differently: Spearman(latency, buffers) within a query has
a median of **+0.341**, and **0 of 59** queries reach +0.8. Buffers count *pages
touched*; latency measures *work done*, and they diverge precisely on the
operator choices this system exists to make.

Training on them would trade the current label's noise for **bias**, and bias
does not average out with more data.

Two uses survive, both from the stability rather than the correlation:

- [ ] **as a per-comparison noise floor** — two candidates with identical buffer
      counts did the same page work, so any latency difference between them is
      noise. Free: it is already in the log. §2.18 measures noise globally; this
      would measure it per pair
- [x] ~~**as features**~~ — **blocked, and checked before writing it down.**
      Buffer counts come from `EXPLAIN ANALYZE`. The production path plans every
      candidate with `analyze=False` and executes only the winner (§2.6), so
      `get_plan(..., analyze=False)["buffers"]` is `None` at serving and
      populated in training. A feature with that shape is train/serve skew by
      construction — the model would fit a signal that is always absent when it
      matters. Verified against a live planner, not reasoned about

### ~~P15. Parameterised queries and the plan cache~~ — **already built, and safety measured**

`query_fingerprint` applies `query_template()`, so `city = 'Mumbai'` and
`city = 'Singapore'` hash to the same id, and `decision_cache` is keyed on it.
Choose once, serve the template — verified directly, both give
`adhoc:e0e00d23...`.

The entry did not ask the more interesting question: **should** they share?
Measured on a template with 63.2% headroom, across a 7x selectivity span:

| literal | rows | own best | served the shared plan | penalty |
|---|---|---|---|---|
| Mumbai | 15,151 | 93.7 | 96.7 | +3.2% |
| Delhi | 6,267 | 80.7 | 86.9 | +7.6% |
| Bangalore | 4,873 | 85.5 | 74.0 | **−13.5%** |
| Singapore | 2,245 | 74.2 | 79.3 | +6.8% |

Two literals nominally have different winning hints — but Bangalore, the one
that differs, comes out **13.5% faster** on the shared plan than on its own
"best". A plan measured as worse cannot be 13.5% faster; the difference is below
the §2.18 noise floor of 18.6%, and so is every other row.

Safe at this scale. Written up as §2.62.

- [ ] **untested at wider spans.** 7x is the whole range this schema offers on
      `city`. A workload where one literal matches 10 rows and another 10
      million is a different question, and the mechanism that would break — a
      plan chosen for a large filter served to a tiny one — is real even though
      it does not fire here

### ~~P16. Cross-database transfer~~ — **measured, and it works**

Both directions, **zero** overlapping tables in each (so every per-table feature
slot reads zero and only the schema-agnostic half is doing anything):

| carried model | target | headroom captured | local model |
|---|---|---|---|
| synthetic (6 tables, 755k rows) | JOB/IMDB (21 tables, 73M rows) | **97.0%** | 98.4% |
| JOB/IMDB | synthetic | 44.1% | 69.3% |

Cost argmin captures 12.8% and 12.1% on the same two, and a random pick is
*negative* on both.

The synthetic model lands **within 1.4 points** of the model trained on JOB
itself. The reverse is weaker because JOB's headroom is enormous and
concentrated (116 s native against a 30 s oracle) — avoiding disasters captures
most of it, and generic structural features suffice for that.

**Latency captured is not accuracy**: the carried models pick the exact best
candidate on 10/30 and 5/65 queries. Good at avoiding bad plans, mediocre at
finding the best — §2.43's finding by another route. Written up as §2.64.

### ~~P17. Teach the regression guard *what* to block~~ — **already built**
`is_family_blocked`, `allowed_hints` and `blocked_families` exist, are wired into
`planner.plan_query`, and are tested. The roadmap entry was stale.

**What was actually wrong**: `app.benchmark` built its candidate list straight
from `candidate_hints` and never applied the family filter, so with the guard on
it measured a **wider** action space than production serves. `--protocol
compare` runs guardless and is unaffected; `--protocol deploy` runs with it on,
and that figure was taken over candidates the server would have dropped. Fixed,
with a test that the filter stays conditional on `use_guard` — filtering under
`compare` would put the guard's state back into a protocol built to exclude it
(§2.30).

### ~~P18. `DisableIndex` for the unused-index advisor~~ — **done**

`app/dbadvisor/index_simulation.py`. For each index `indexes.unused()` reports,
it plans every workload query twice — once normally, once with
`DisableIndex(alias index)` — and reports which plans get materially worse and
by how much. No lock, no catalog write, nothing executed: `DROP INDEX` in a
rolled-back transaction would also work and takes an ACCESS EXCLUSIVE lock, which
is not something an advisory tool should do to a production table.

Verified against a *used* index, since this schema has no unused ones: dropping
`orders_pkey` would cost 2 of 39 queries (worst 1.21x), `users_pkey` 4 of 31.

Three distinctions the verdict makes that a scan counter cannot:

- **"no query tested it"** vs **"every query was fine without it"** — only the
  second is evidence, and the first says so explicitly;
- **one alias per self-join** — `DisableIndex` names an alias, not a table, so a
  self-join needs one hint per alias or half the query keeps the index;
- **uniqueness** — dropping a unique index changes what the database will
  *accept*, which no amount of plan evidence makes safe. Carried into the
  verdict text.

Wired into `indexes.report()` and the `/dbadvisor` endpoint, with tests that
assert it is actually reached — a module nothing calls is the bug §2.42, §2.50
and §2.53 all are.

### P19. Re-measure the two switched-off families
`INCLUDE_JOIN_METHODS` and `INCLUDE_BUSHY` both bind correctly since §2.20 and
§2.29. §2.27 measured zero wins across 24 queries — **before** the workload
tripled, before the shape group existed, and before PostgreSQL 18.

**Note the trap (§2.52)**: switching a family off removes the evidence needed to
switch it back on. The sweep reads `latest_run_only=True`, so once collection
stops producing a family, no future sweep can price it. Re-measuring requires
deliberately collecting with it on.

---

## Untested, not broken

- **JOB above 7 tables.** The benchmark goes to **17-table joins**; only the
  smallest 30 queries have ever been collected. The candidate sampler has never
  been pointed at the hard end.
- **Cold start.** Before a model exists, selection falls back to the cost
  heuristic — which JOB shows can be *worse than native*.
- **Single-node, single-connection.** Every latency here is measured with no
  concurrency. Nothing is known about behaviour under load.
- **Synthetic skew is synthetic.** `data/schema.sql`'s correlations are clean
  compared to real long tails. The JOB numbers are the external-validity
  evidence; the synthetic ones are a demonstration.

---

## The standing one

**Fifteen-plus silent bugs found, and the rate is not falling.** Seven landed in
a single session (§2.42–2.54), including the two most consequential in the
project's history — a gate calibrated under a policy the server does not run
(§2.43) and a replay that could not see the safety veto (§2.47). Neither produced
an error, a warning, or an implausible number.

The two questions that keep finding them, and should keep being asked:

1. **"Is this code reached?"** — found `pg_stat_statements` never created
   (§2.36), two hint families never re-measured (§2.27), a conformal gate fitted
   every run and consulted on none (§2.42), a swept `gate_mode` the API ignored
   (§2.50), a setting with six lines of help text and no effect (§2.53), and 39%
   of the action space classified as `other` and dropped from every replay
   (§2.56).
2. **"Does this measurement agree with what runs?"** — found the policy (§2.43),
   the action space (§2.50), the candidate pool (§2.47), the promotion yardstick
   (§2.45), the drift window (§2.49), and collection being narrower than serving
   (§2.56).

Checks now exist for the three that are checkable: `models/gate.json` records the
policy, the action space and the moment it was applied, and `app.diagnostics`
compares each against reality. That is a response to the base rate, not an answer
to it.
