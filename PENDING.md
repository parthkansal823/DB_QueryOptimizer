# Pending work

Everything still unfinished, with the measurement that says why it matters.
Ordered by what would change the project's claims most, not by effort.

Closed items are **removed** rather than struck through — what each one found is
written up in `docs/WRITEUP.md`, which is the record. §2.55–2.69 cover the work
that emptied most of this file.

`docs/ROADMAP.md` is the same list with the reasoning; this is the checklist.
`docs/WRITEUP.md` §4 is the honest list of what is still wrong.

Last updated 2026-09-05.

---

## Blocking — numbers that are currently stale

### P1. Finish the JOB re-collection

**Why**: the **99.2%** figure in `docs/JOB_RESULTS.md` is measured against the
*old*, narrower ceiling. §2.56 widened the action space and the ceiling moved
under it — confirmed on the queries collected so far:

| query | old oracle | new oracle | old headroom | new headroom |
|---|---|---|---|---|
| `2b` | 460.9 | **311.1** | 0.6% | **34.0%** |
| `2d` | 571.7 | **395.5** | 2.5% | **40.0%** |
| `5b` | 83.3 *(worse than native)* | **68.2** | none | **9.4%** |

Across the 23 queries with data, **74.6%** of the total latency is now
reachable. `5b` matters most for honesty: §2.55 describes it as a query where no
candidate beats PostgreSQL. That was true of the narrow action space.

**Stopped deliberately at 14 of 30.** The wider budget gives the `17*` family
114 candidates instead of 30: `17d` took **64 minutes** and `17e` was still
running after **2h21m**. Four more of them would have been 10+ hours, and it was
blocking every measurement that needs a quiet database for clean latencies —
§2.68's paired repetitions among them.

```bash
DATABASE_URL=...job INCLUDE_SCAN_METHODS=1 \
  python -m app.collect_data_job --n 30 --reps 2 --timeout-ms 30000
```

30 s is a real trade rather than a free speed-up: `17c` runs 21.6 s natively, so
that cap admits only candidates within ~1.4x of native and the model stops
seeing moderately-bad plans for those queries. The narrow-space rows from the
earlier complete run still carry them.

- [ ] finish the collection (overnight, or at the tighter timeout)
- [ ] re-train and compare against 99.2%
- [ ] update `docs/JOB_RESULTS.md`, `docs/WRITEUP.md` §2.55, `README.md`

---

## High value — attacks a measured, binding constraint

### P2. Fix the ranking on unseen queries

**Why**: the model captures ~43% of held-out headroom, so more than half is
still on the table — and §2.68 is the sharp version. On six six-table queries
carrying **23.6%** headroom, the model was allowed to act twice and lost both
times (1.017x and 1.095x native, paired over seven repetitions), on queries it
had **trained on**. The guard stopped the other four. Neither losing pick came
from an optional family; both were `operator_toggle` and `cardinality`, always
on. The action space is not the constraint and the gate is not the constraint.

§2.35 measured the model *uniformly* inaccurate rather than overfitted — 2.51x
typical multiplicative error on trained queries, 2.66x on held-out — and the
current 1.27x is better and still the same order as the speedups worth chasing.

§2.43 sharpens it: uniform error mostly cancels in a *comparison* and not at all
in a *prediction*, which is why `pairwise_rank` beats every point-prediction
policy by 36 points. The work is to spend the model's capacity on comparisons.

§2.69 found where a third of it was going instead. `app.feature_audit` reports
**23 of 67** features taking one value per query — every `*_present` and
`*_occurrences` slot, `num_joins`, `num_tables`, and `has_hint`, which §2.50
identified as useless and which still collects 255 splits. A feature constant
across a query's candidates cannot change a comparison, and the trees spend
**6.8%** of their splits on that group. On JOB's 21-table schema it is **72 of
139** — more than half the vector — so the effect grows with schema width.

- [ ] **give the ranker the diagnostics the ensemble has.** `uncertainty_scale`,
      the conformal bound, the calibration sweep, the promotion score — every
      one describes the ensemble. The component that actually chooses plans has
      none of them, and §2.63 is what that costs: a 0.610x prediction on a plan
      that ran at 1.20x, with nothing able to flag it
- [ ] **promote the 44-feature model, or find out why not.** Dropping all 23
      improved held-out capture on **11 of 16** paired splits — mean **+8.3
      points**, signed-rank p = 0.026, eleven gains averaging +12.8 against five
      losses averaging −1.5 — and typical error on 13 of 16. It is a model
      change, so it goes through `app.retrain`'s champion/challenger comparison
      like any other; §2.63 is what happens when a better mean hides a worse
      worst case
- [ ] buffer counts as a **per-comparison noise floor** — not as a label
      (§2.58: Spearman 0.341 with latency) and not as a feature (train/serve
      skew: they come from `EXPLAIN ANALYZE` and serving plans with
      `analyze=False`). Two candidates with identical buffer counts did the same
      page work, so any latency difference between them is noise. Free — it is
      already in the log

### P3. Ship the extended statistics, or decide not to

Measured (§2.57): three `CREATE STATISTICS` statements and one `ANALYZE` remove
**98%** of the opportunity on the queries this schema was built around.

| arm | native | oracle | headroom |
|---|---|---|---|
| statistics ON | 595.9 | 567.4 | **0.4%** [0.2, 4.8] |
| statistics OFF | 602.0 | 481.4 | **18.6%** [14.2, 20.0] |

Ranges disjoint. Native is unchanged — PostgreSQL's own plan becomes the good
one and the oracle rises to meet it.

Left off deliberately: the deployed model and the benchmark were both measured
without them, so turning them on needs a re-collect.

- [ ] decide whether `data/schema.sql` ships with the statistics created
- [ ] if yes: re-collect, re-train, and expect the headline to **fall** — that
      is the metric working, not breaking

---

## Known-wrong, cause understood, fix not attempted

### P4. The benchmark cannot resolve the differences being argued about

Three compounding problems, all measured: repeated executions of the same plan
vary **18.6%** (§2.18); the benchmark mutates the guard's state, so two runs of
an identical gate gave 15.6% and 7.2% (§2.30); two independent 3-run
measurements of an identical configuration gave medians **8 points apart**
(§2.32). The latest run's own floor is **7.7 points**.

The tooling respects this now — `MIN_POLICY_SWITCH_POINTS`, median-of-draws,
the printed floor — but the instrument itself has not improved.

**And the train-time report is worse.** §2.66 swept the query-level split over
ten seeds, same rows and same model class:

| statistic | across ten splits |
|---|---|
| conformal coverage | 62.5% to 92.6%, median **75.5%** |
| headroom captured | 24.2% to **89.1%** — a 64.9-point range |

Different cause: the benchmark's problem is measurement noise, this one is
sampling, and 16 held-out queries out of 65 is the sample doing the damage.
`app.coverage_sweep` reports the range in about two minutes, so quoting one
split is now a choice.

- [ ] more repetitions, or a paired design that cancels between-run drift
- [ ] until then: no claim resting on a benchmark difference under ~8 points,
      and none at all on a single split's held-out capture

### P5. The unbiased sample is small and hand-picked

`app.benchmark` leads with the figures nobody selected for and names the gap:
the selection effect is **+14.8 points**, up from +8.3 once six unfiltered
six-table queries were added.

But the curated group is unbiased *and* hand-written, so it has its own
selection story.

- [ ] generate a workload with the ≥8% headroom filter removed, and compare —
      that would give an unbiased estimate over *generated* queries too

More queries is also the only real fix for §2.66's 64.9-point split range, so
this item is worth more than its position suggests.

---

## Not started

- **A marker for bushy join orders.** §2.67 priced every optional hint family
  on both databases; `INCLUDE_BUSHY` was the one it could not, because bushy
  hints are `Leading(((a b) (c d)))` and classify as `join_order`, so they are
  indistinguishable from linear ones in the log. It needs a marker before the
  sweep can say anything about it.
- **Features from the query beyond predicate shape.** §2.60 added
  `max_same_table_equalities` and two counts; the query text carries more.
- **Wider selectivity spans for the plan cache.** §2.62 measured template
  sharing safe across a 7x span, which is the whole range this schema offers on
  `city`. A workload where one literal matches 10 rows and another 10 million is
  a different question, and the mechanism that would break — a plan chosen for a
  large filter served to a tiny one — is real even though it does not fire here.

---

## Untested, not broken

- **JOB above 7 tables.** The benchmark goes to **17-table joins**; only the
  smallest 30 queries have ever been collected.
- **Cold start.** Before a model exists, selection falls back to the cost
  heuristic — which JOB shows can be *worse than native*.
- **Single-node, single-connection.** Every latency here is measured with no
  concurrency. Nothing is known about behaviour under load.
- **Synthetic skew is synthetic.** §2.64's transfer result is the strongest
  external-validity evidence there is: a model trained on the 6-table synthetic
  schema captures **97.0%** of JOB's headroom against a locally-trained model's
  98.4%, with zero overlapping tables.

---

## The standing one

**Seventeen-plus silent bugs found, and the rate is not falling.** Every failure
catalogued in section 2 produced a plausible number as its only symptom.

The two questions that keep finding them, and should keep being asked:

1. **"Is this code reached?"** — found `pg_stat_statements` never created
   (§2.36), two hint families never re-measured (§2.27), a conformal gate fitted
   every run and consulted on none (§2.42), a swept `gate_mode` the API ignored
   (§2.50), a setting with six lines of help text and no effect (§2.53), and 39%
   of the action space classified as `other` and dropped from every replay
   (§2.56), and the calibration sweep scoring 117 of 513 candidates because
   running the benchmark hid the collection run behind it (§2.65).
2. **"Does this measurement agree with what runs?"** — found the policy (§2.43),
   the action space (§2.50), the candidate pool (§2.47), the promotion yardstick
   (§2.45), the drift window (§2.49), collection being narrower than serving
   (§2.56), and the conformal coverage check reading uncalibrated means while
   the gate is fitted and consulted on calibrated ones (§2.66 — real, and worth
   under a point, which is itself worth knowing).

A third has started earning its keep: **"is this worth building?"**, asked
*before* building. §2.60's feature was measured monotone (16.9% → 25.3% →
31.4%) before a line was written; §2.61's was measured at +0.04 and never built;
§2.58's label was rejected on a 0.341 correlation; per-join-width models were
rejected on 96 rows across 3 queries. The cost of asking first is one script
each, and it is the cheapest thing in this file.

**And §2.69 is the refinement it needed.** §2.60 asked "is this worth building?"
properly, measured `max_same_table_equalities` monotone against headroom
(16.9% → 25.3% → 31.4%), and built it on that evidence. The measurement was
real and the *axis* was wrong: it showed the feature predicts which **queries**
have headroom, and the ranker compares **plans within one query**. Ablated, all
three of §2.60's features changed the pick on 0 of 16 queries.

So the question has a second half: **"measured against what the component
actually decides?"** For anything feeding `pairwise_rank` that is now
`app.feature_audit`, which is one command.

Checks now exist for the ones that are checkable: `models/gate.json` records
the policy, the action space and the moment it was applied, `app.diagnostics`
compares each against reality, `app.coverage_sweep` reports a held-out figure as
a range over splits rather than a number, and `app.feature_audit` names every
feature that cannot separate the things being compared.
