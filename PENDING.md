# Pending work

Everything still unfinished, with the measurement that says why it matters.
Ordered by what would change the project's claims most, not by effort.

Closed items are **removed** rather than struck through — what each one found is
written up in `docs/WRITEUP.md`, which is the record. §2.55–2.88 cover the work
that emptied most of this file.

`docs/ROADMAP.md` is the same list with the reasoning; this is the checklist.
`docs/WRITEUP.md` §4 is the honest list of what is still wrong.

Last updated 2026-09-06.

---

## High value — attacks a measured, binding constraint

### P1. Fix the ranking on unseen queries

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

§2.69–2.71 spent this budget and moved the constraint. What was found:

| tried | result |
|---|---|
| drop 23 within-query-constant features | one of them (`has_hint`) is the whole +8.3 (§2.70) |
| drop baseline pairs from ranker training | **−4.2 points** — the argument was good, the data was not |
| gate on the ranker's own margin | Spearman **+0.021**, worst case in the *widest* bucket |
| `borda`: aggregate ensemble + ranker ranks | **63.0% vs 52.1%**, worst case 2.22x → 1.64x |

**The gate is now the binding constraint, and it is the last one standing.**
`borda` is the better selector ungated and `app.autotune` puts it *second*
(+2.9% against `pairwise_rank`'s +4.3%) because it deviates on 44% of queries
instead of 81%. The gate thresholds on the ensemble's point prediction for
whichever candidate is put forward, and borda's picks are compromises — rated
well by both selectors, extreme by neither. A gate on predicted gain
systematically under-authorises exactly the candidates aggregation exists to
find.

- [ ] **decide `min_relative_gain`.** §2.79: the sweep recommends **0**, which
      makes `required = 1 - gain * caution` equal 1.0 for every caution — so
      the caution mechanism is inert on the expected-win test, including the
      regression guard's 2.0 for a query it has no history for. Left as the
      sweep measured it (0.0 nets +4.2% against 0.02's +2.9%, not a tie) with a
      warning printed, because overriding a measurement on judgment is the thing
      this project does not do. But the sweep scores 16 held-out queries and
      caution exists for the queries it never saw, which is an argument it
      structurally cannot make. **This is a call for a person.**
- [ ] **make the gate read something other than a point estimate.** This is
      §2.63's item and everything queues behind it: the 44-feature model
      (rejected, worse tail), `borda` (better selector, under-authorised),
      §2.66's bound that covers 75% while advertising 90%, and now §2.79's
      disagreement term — a real risk signal (13% vs 22% regression, 1.64x vs
      3.22x worst) that is shipped off because it multiplies an inert quantity.
      Three of the four gate knobs interact and the grid treats them as
      independent
- [ ] **promote the 44-feature model, or find out why not.** *Found out*: it
      captures 46.5% → 67.6% on seed 42 and `app.retrain` **rejected** it —
      "worst pick is 3.31x the best available against the champion's 1.41x".
      Second challenger in a row rejected for the same trade. Needs the item
      above, not another model
- [x] ~~buffer counts as a **per-comparison noise floor**~~ — built, §2.88.
      `app.noise --from-buffers` derives a floor per query from pairs of plans
      that did identical page work: **306 groups, 1,290 pairs, at least one on
      every query**, median disagreement 12.7% and worst **110.5%**. The global
      18.6% is wrong in both directions — `gen_27_5w`'s own floor is 129.7% and
      `shape_group_by_category`'s is 5.9%.
- [ ] **consume it.** `stats.classify` still grades every query against one
      workload-wide `recommended_material_fraction`. Switching to the query's
      own floor changes what `held_missed` and `deviated_win` *mean*, so it
      wants a re-run benchmark beside it rather than being folded in with
      everything else

---

## Known-wrong, cause understood, fix not attempted

### P2. The benchmark cannot resolve the differences being argued about

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

`--protocol paired` is the paired design: it re-times native, served and oracle
**alternately** per query and takes medians, so drift is common-mode instead of
landing on one arm. It also re-times the oracle, which is a `min` over noisy
draws and therefore biased low even when every timing is unbiased — that
inflates the headroom denominator and deflates every capture figure computed
from it. Costs three executions per repetition per query rather than a whole
extra workload pass.

- [x] ~~re-measure the synthetic headline under `--protocol paired`~~ — done,
      §2.85. Headroom **1,304 → 448 ms** and the saving **797 → 215 ms** against
      the same code one pass instead of five: three quarters of the measured
      benefit was noise. `README.md` republished.
- [ ] run it more than once. Every figure in §2.85 is a single run and
      `paired` says so. The direction is structural — the `min` bias is
      one-sided, so re-timing can only lower the reported headroom — but the
      magnitude is not pinned.
- [ ] **more queries — the one fix nothing else substitutes for.** §2.66's
      64.9-point split range is sampling, not noise: 16 held out of 65. Paired
      timing cannot touch it and neither can more repetitions. §2.72 makes it
      cheap to say what a bigger workload would cost, though — only **6 of 40**
      generated queries clear 8% headroom, so a workload with 50 useful queries
      means generating and vetting on the order of 350
- [ ] standing rule: no claim resting on a benchmark difference under ~8 points,
      and none at all on a single split's held-out capture

---

## Not started

- **Collect with the unscored families on, then price them.** §2.75 and §2.84
  between them took the sweep axis to all **seven** optional families —
  `bushy` and `disable_index` are now separable from their parent families —
  and derived it from the data, so it reports five as **not scored**: the log
  has no rows for any of them. Collecting is the only way to turn that into an
  answer, and it is now a cost rather than a question:

  ```bash
  INCLUDE_JOIN_METHODS=1 INCLUDE_MEMOIZE=1 INCLUDE_PARALLEL=1     INCLUDE_BUSHY=1 INCLUDE_DISABLE_INDEX=1 python -m app.collect_data
  python -m app.autotune
  ```

  Budget for it: each family widens what every query has to execute.
- **Features from the query beyond predicate shape** — but read §2.69 first.
  §2.60's three were measured to change **zero** ranker decisions, because
  anything derived from the SQL text takes one value per query and the ranker
  compares within a query. A query-text feature can only help the *ensemble*
  (which predicts a level) or a component that decides **whether to deviate at
  all** — which is a real use, and not the one they were built for.
  `app.feature_audit` is the check to run before writing the next one.
- **Wider selectivity spans for the plan cache.** §2.62 measured template
  sharing safe across a 7x span, which is the whole range this schema offers on
  `city`. A workload where one literal matches 10 rows and another 10 million is
  a different question, and the mechanism that would break — a plan chosen for a
  large filter served to a tiny one — is real even though it does not fire here.

---

## Untested, not broken

- **JOB above 7 tables, and above the GEQO threshold.** The benchmark's largest
  queries are the `29*` family at **17 tables**; only the smallest 30 have ever
  been collected, and the largest of those joins seven. §2.81 counted it:
  **20 of 113** JOB queries reach `geqo_threshold = 12`, so the genetic
  optimiser has never run on a single query measured in this project — on
  either database. Everything known here is about the dynamic-programming
  planner.
  (Note the naming trap: `17c` is a query *name*, not a table count. It joins
  seven.)
- ~~**Cold start.**~~ Tested, and it *was* broken — §2.82. A fresh install
  deviated on **every** query and ran 1.040x native with a 5.32x worst case.
  Fixed by serving native until a model exists; the obvious repair (deviate
  only when the cost estimate says cheaper) measured **worse**, 1.109x with
  83% of its deviations regressing.
- ~~**Single-node, single-connection.**~~ Partly tested — §2.87. The regression
  guard survives concurrent refresh (**3.1M reads, zero errors**), and it is
  safe because `refresh` rebinds rather than mutates, which nothing had written
  down; that is now a docstring and a test. Planning overhead holds at
  **41–47%** of execution from 1 to 8 concurrent planners — both degrade ~2.5x
  together, so the economic case does not worsen under load.
  **Still untested**: the serving path end to end (FastAPI, the connection
  pool, the decision cache under contention), and anything beyond one machine.
- ~~**Synthetic skew is synthetic.**~~ Re-run — §2.86, and the claim did not
  survive. The carried model captures **−5.3%** on the wide action space
  (97.0% before, against a narrow ceiling), serving JOB 4% *slower* than doing
  nothing, against a local model's 96.5%. It still beats random by 22 points, so
  the structural features rank well above chance and not above native.
  **The open question it leaves**: does a carried model make local training
  converge faster than a cold start? That is the useful version of transfer and
  it has never been asked.

---

## The standing one

**Thirty-plus silent bugs found, and the rate is not falling.** Every failure
catalogued in section 2 produced a plausible number as its only symptom.

The two questions that keep finding them, and should keep being asked:

1. **"Is this code reached?"** — found `pg_stat_statements` never created
   (§2.36), two hint families never re-measured (§2.27), a conformal gate fitted
   every run and consulted on none (§2.42), a swept `gate_mode` the API ignored
   (§2.50), a setting with six lines of help text and no effect (§2.53), and 39%
   of the action space classified as `other` and dropped from every replay
   (§2.56), the calibration sweep scoring 117 of 513 candidates because
   running the benchmark hid the collection run behind it (§2.65), and
   `INCLUDE_DISABLE_INDEX` generating **zero** candidates because the catalog
   map its family needs was gated on a *different* flag (§2.74), and the sweep's
   family axis covering six of the eight families `hint_family` names, so
   `memoize` and `parallel` were dropped from every arm (§2.75), and a
   recommended threshold of `min_relative_gain = 0` making the whole caution
   mechanism inert on the test that decides whether a win is worth taking —
   including the guard's 2.0 for an unseen query — with the module's own
   docstring making exactly that argument about the *other* knob (§2.79).
2. **"Does this measurement agree with what runs?"** — found the policy (§2.43),
   the action space (§2.50), the candidate pool (§2.47), the promotion yardstick
   (§2.45), the drift window (§2.49), collection being narrower than serving
   (§2.56), and the conformal coverage check reading uncalibrated means while
   the gate is fitted and consulted on calibrated ones (§2.66 — real, and worth
   under a point, which is itself worth knowing), `has_hint` separating the 65
   baseline rows from 2,257 candidate rows in training while being constant at
   serving (§2.70 — 255 splits, and deleting it was the largest single model
   improvement found this year), and a gate calibrated for the *previous* model
   passing both drift checks because the only thing recorded about the model was
   a path that had not changed (§2.73), and collection skipping the prefilter
   serving runs, so **14.4%** of logged candidates are plans byte-identical to
   native and one of them is the oracle on 11 of 65 queries — in all three
   functions that compute a best-available, fixed in one of them first (§2.76),
   and a hint family blocked on a query being unable to produce the evidence
   that would unblock it — the shadow path applied the family filter, and the
   family query ignored shadow rows, so either half alone left the ratchet
   intact (§2.77), and a comment asserting that two definitions of query
   identity stay in sync when they disagree on two of three cases (§2.78 — a
   wrong comment is worse than none, because it is the reason nobody checks),
   and the deterministic measurement session pinning `geqo_seed` away from the
   value serving uses, on a premise about that setting that turned out to be
   false (§2.81), a cold-start path that deviated on **every** query with no
   comparison against native at all, contradicting the docstring it sat under
   (§2.82), and `app.benchmark` handing the model candidates carrying disabled
   nodes that `plan_query` drops — so 36 of 43 vetoes had both gate conditions
   satisfied and the query lost its deviation entirely (§2.83).

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

A fourth question, earned the hard way in §2.70: **"which population is this
computed over?"** `app.feature_audit` and `app.train --drop-constant-features`
answered "is this feature constant?" over different row sets, written an hour
apart — candidates only, and candidates plus baselines — and the one that
*acted* was blind to the only feature worth acting on. They share
`features.within_query_variance` now.

A fifth, and it is the one this stretch kept needing: **"is the *tool* wrong,
or the thing it measures?"** §2.83's deviation gap read as a mis-calibrated
gate and was a benchmark handing out candidates the server never offers. §2.76's
prefilter looked too aggressive and was correct, with the log's oracle
contaminated instead. §2.86's transfer collapse was not the model changing —
nothing about it changed — but the ceiling it was scored against. Three times
in one night the instrument was the defect, and each time the first reading
blamed the system.

Checks now exist for the ones that are checkable: `models/gate.json` records
the policy, the action space and the moment it was applied; `app.diagnostics`
compares each against reality; `app.coverage_sweep` reports a held-out figure as
a range over splits rather than a number; `app.feature_audit` names every
feature that cannot separate the things being compared;
`app.selection_effect` says how much of the headline is the workload having been
chosen; `app.benchmark --protocol paired` re-times the arms alternately, so a
regression has to survive the noise before it gets reported; and
`app.noise --from-buffers` gives every query its own noise floor for free, from
pairs of plans that provably did the same page work.
