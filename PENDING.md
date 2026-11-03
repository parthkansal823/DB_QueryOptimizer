# Pending work

Everything still unfinished, with the measurement that says why it matters.
Ordered by what would change the project's claims most, not by effort.

Closed items are **removed** rather than struck through — what each one found is
written up in `docs/WRITEUP.md`, which is the record. §2.55–2.94 cover the work
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

- **Confirm the family sweep by execution, not replay.** §2.94 swept all 128
  action spaces under the served policy and recommends the narrow one: not one
  of the seven optional families earns its place, `bushy` is bit-identical to
  omitting it, and `scan_method`/`parallel`/`disable_index` clear the safety
  bounds in none of the 128 arms. The winning arm reads **+8.7% net** on 16
  held-out queries.

  Two reasons that is not finished. It is a **replay** — logged rows scored
  against a model, which is the thing §2.83 found handing the model candidates
  the server never offers. And the whole spread of the thirteen surviving arms
  is **2.7 points**, against §2.32's 8-point between-run drift, so the replay
  cannot separate any of them from each other.

  ```bash
  python -m app.benchmark --protocol paired
  ```

  What is already resolvable without that: `bushy` earns exactly zero (four
  exact ties, not a small difference), and three families fail the bounds
  outright. Those do not need re-measuring.

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

---

## The standing one

**Thirty-plus silent bugs found, and the rate is not falling.** Every failure
catalogued in section 2 produced a plausible number as its only symptom.

The audit questions that keep finding them — with the sections that earned each
one — now live in `docs/WRITEUP.md` under the section-2 summary rather than
here, because they are a record of what was learned, not work outstanding:

1. **Is this code reached?** (§2.36, 2.42, 2.50, 2.56, 2.65, 2.74, 2.75, 2.79,
   2.90, 2.91)
2. **Does this measurement agree with what runs?** (§2.43, 2.45, 2.47, 2.49,
   2.66, 2.70, 2.73, 2.76, 2.77, 2.78, 2.81, 2.82, 2.83, 2.89)
3. **Is this worth building?**, asked *before* building (§2.58, 2.60, 2.61)
   — and its second half, **measured against what the component actually
   decides?** (§2.69)
4. **Which population is this computed over?** (§2.70) — with the corollary
   that when the population is *collected*, **the budget and the population
   must be the same choice** (§2.92)
5. **At what granularity does this rule resolve conflicts?** (§2.93) — "latest
   run wins" was a decision about *runs* and is unavoidably also a decision
   about every family nested inside one, which makes a narrow collection
   destructive rather than additive
6. **Is the *tool* wrong, or the thing it measures?** (§2.76, 2.83, 2.86)
7. **Which language reads this string?** (§2.91)

The checks that exist for the ones that are checkable: `models/gate.json`
records the policy, the action space and the moment it was applied;
`app.diagnostics` compares each against reality; `app.coverage_sweep` reports a
held-out figure as a range over splits; `app.feature_audit` names every feature
that cannot separate the things being compared; `app.selection_effect` says how
much of the headline is the workload having been chosen; `app.benchmark
--protocol paired` re-times the arms alternately; `app.noise --from-buffers`
gives every query its own noise floor; `app/env.py` holds one parser per type,
enforced by an AST walk over every module; and `app.collect_data --held-out`
aims a collection run at the queries the sweep can score.
